# 4단계 수정본
# =========================================================
# 4단계 V4 최종 수정본 (DRM/xlwings + hard timeout + resume)
# PPID 필터를 통과한 wafer의 chip-level UFI 조회
#
# 성능 개선:
#   1. wafer-level UFI의 tkout_time 날짜별로 1일씩 조회
#   2. line + process + UFI item + 측정일 단위로 그룹화
#   3. Root LOT을 기본 20개씩 묶어서 조회
#   4. Impala 15분 timeout 발생 시 해당 배치만 자동으로 절반 분할
#   5. root_lot_id + wafer_id로 최종 정확하게 필터링
#   6. UFI item은 항상 각각 별도로 조회/저장
#   7. 배치 cache, split marker, target coverage 저장
#
# 5단계 호환 변수:
#   ufi_chip_filtered_all
# =========================================================
from pathlib import Path
from datetime import datetime

import hashlib
import pickle
import re
import subprocess
import sys
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import bigdataquery as bdq


STAGE4_CODE_VERSION = 'STAGE4_V4_DRM_HARDTIMEOUT_RESUME_V0901'
print(f'\n[CODE VERSION] {STAGE4_CODE_VERSION}')


# =========================================================
# 1. 사용자 설정
# =========================================================
if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = Path.cwd() / 'UFI_OVL_OUTPUT'

# 최초 배치 크기
# timeout이면 20 -> 10 -> 5 -> 2 -> 1 순으로 자동 분할된다.
INITIAL_ROOT_LOT_BATCH_SIZE = 20
MIN_ROOT_LOT_BATCH_SIZE = 1

# root lot 조건과 함께 wafer_id 조건도 DB에 전달한다.
# API가 wafer_id 조건을 지원하지 않으면 자동으로 조건 없이 재시도한다.
USE_WAFER_ID_QUERY_FILTER = True

# 1단계 wafer UFI와 동일한 subitem만 조회한다.
UFI_SUBITEM_ID = 'A_CSUM_PERCENT'

# bdq.getData가 Impala 오류조차 반환하지 않고 멈추는 경우를 막기 위해
# 별도 Python process에서 실행하고 이 시간이 지나면 process를 종료한다.
CLIENT_HARD_TIMEOUT_SEC = 18 * 60
QUERY_HEARTBEAT_SEC = 60

# timeout 이외의 일시적 오류 재시도 횟수
QUERY_RETRY = 2
QUERY_RETRY_WAIT_SEC = 3
QUERY_PAUSE_SEC = 0

SAVE_COMBINED_PICKLE = True
SAVE_QUERY_CACHE = True

CACHE_VERSION = 'DAILY_ADAPTIVE_V1'


# =========================================================
# 2. 공통 함수
# =========================================================
def find_column(df, candidates, required=True):
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise KeyError(
            '다음 column 중 사용할 수 있는 column이 없습니다.\n'
            f'{candidates}\n\n'
            f'현재 column:\n{df.columns.tolist()}'
        )

    return None


def categorical_to_string(df):
    result = df.copy()

    for column in result.columns:
        if isinstance(result[column].dtype, pd.CategoricalDtype):
            result[column] = result[column].astype('string')

    return result


def clean_id_series(series, upper=False):
    result = (
        series
        .astype('string')
        .str.strip()
        .str.replace(r'\.0$', '', regex=True)
    )

    if upper:
        result = result.str.upper()

    return result


def clean_text_series(series):
    return (
        series
        .astype('string')
        .str.strip()
        .str.upper()
    )


def normalize_datetime_series(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype('string')

    return pd.to_datetime(series, errors='coerce')


def safe_token(value):
    return re.sub(
        r'[^A-Za-z0-9_.-]+',
        '_',
        str(value)
    )


def split_item_value(value):
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        raw_values = list(value)
    else:
        raw_values = [value]

    item_ids = []

    for raw_value in raw_values:
        if pd.isna(raw_value):
            continue

        item_ids.extend(
            item.strip().upper()
            for item in re.split(r'[,;|]+', str(raw_value))
            if item.strip()
        )

    return sorted(set(item_ids))


def split_batches(values, batch_size):
    values = list(values)

    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def is_timeout_or_resource_error(error):
    text = str(error).lower()

    patterns = [
        'execution time limit',
        'expired due to',
        'query expired',
        'timeout',
        'timed out',
        'memory limit exceeded',
        'resource limit',
    ]

    return any(pattern in text for pattern in patterns)


def found_wafer_count(df):
    required = {'root_lot_id_key', 'wafer_id_key'}

    if df.empty or not required.issubset(df.columns):
        return 0

    return (
        df[['root_lot_id_key', 'wafer_id_key']]
        .drop_duplicates()
        .shape[0]
    )


def filter_ufi_subitem(df):
    """
    기존 cache와 신규 조회 결과 모두 A_CSUM_PERCENT만 유지한다.
    subitem column 자체가 없으면 원본을 유지한다.
    """
    if df is None or df.empty:
        return df

    subitem_column = next(
        (
            column
            for column in df.columns
            if str(column).strip().lower() == 'subitem_id'
        ),
        None,
    )

    if subitem_column is None:
        return df

    subitem_key = clean_text_series(df[subitem_column])

    return df[
        subitem_key.eq(str(UFI_SUBITEM_ID).strip().upper())
    ].copy()


def make_unique_headers(raw_headers):
    headers = []
    counts = {}

    for column_no, raw_header in enumerate(raw_headers, start=1):
        if raw_header is None or str(raw_header).strip() == '':
            base_name = f'Unnamed: {column_no - 1}'
        else:
            base_name = str(raw_header).strip()

        duplicate_no = counts.get(base_name, 0)
        counts[base_name] = duplicate_no + 1

        header = (
            base_name
            if duplicate_no == 0
            else f'{base_name}.{duplicate_no}'
        )
        headers.append(header)

    return headers


def read_excel_sheet_robust(excel_path, sheet_name):
    """
    일반 xlsx는 openpyxl로 읽고, 회사 DRM/암호화 파일처럼
    Excel에서는 열리지만 ZIP 기반 reader가 읽지 못하는 파일은
    xlwings(Excel COM)로 읽는다.
    """
    excel_path = Path(excel_path)
    openpyxl_error = None

    try:
        data = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            dtype=str,
            engine='openpyxl',
        )
        return data, 'OPENPYXL'

    except Exception as error:
        openpyxl_error = error

    try:
        import xlwings as xw
    except Exception as xlwings_import_error:
        raise RuntimeError(
            f'openpyxl 읽기 실패: {type(openpyxl_error).__name__}: '
            f'{openpyxl_error}\n'
            f'xlwings import 실패: '
            f'{type(xlwings_import_error).__name__}: '
            f'{xlwings_import_error}'
        ) from xlwings_import_error

    app = None
    workbook = None

    try:
        app = xw.App(visible=False, add_book=False)
        app.display_alerts = False
        app.screen_updating = False

        workbook = app.books.open(
            str(excel_path.resolve()),
            update_links=False,
            read_only=True,
            ignore_read_only_recommended=True,
        )

        sheet_names = [sheet.name for sheet in workbook.sheets]

        if sheet_name not in sheet_names:
            raise KeyError(
                f'{sheet_name} sheet가 없습니다. '
                f'현재 sheet: {sheet_names}'
            )

        values = workbook.sheets[sheet_name].used_range.value

        if values is None:
            return pd.DataFrame(), 'XLWINGS'

        if not isinstance(values, list):
            values = [[values]]
        elif values and not isinstance(values[0], list):
            values = [values]

        if not values:
            return pd.DataFrame(), 'XLWINGS'

        headers = make_unique_headers(values[0])
        data = pd.DataFrame(values[1:], columns=headers)
        data = data.dropna(how='all').reset_index(drop=True)

        print(
            f'Excel COM(xlwings)으로 읽음: '
            f'{excel_path.name} / {sheet_name}'
        )

        return data, 'XLWINGS'

    except Exception as xlwings_error:
        raise RuntimeError(
            f'openpyxl 읽기 실패: {type(openpyxl_error).__name__}: '
            f'{openpyxl_error}\n'
            f'xlwings 읽기 실패: {type(xlwings_error).__name__}: '
            f'{xlwings_error}'
        ) from xlwings_error

    finally:
        if workbook is not None:
            try:
                workbook.close()
            except Exception:
                pass

        if app is not None:
            try:
                app.quit()
            except Exception:
                pass


# =========================================================
# 3. PPID 필터를 통과한 wafer 목록 가져오기
# =========================================================
if (
    'matched_ppid' in globals()
    and isinstance(matched_ppid, pd.DataFrame)
    and not matched_ppid.empty
):
    ufi_targets = matched_ppid.copy()
    print('현재 메모리의 matched_ppid를 사용합니다.')

else:
    ppid_files = sorted(
        [
            path
            for path in Path(OUTPUT_DIR).glob('PPID_FILTERED__*.xlsx')
            if path.is_file()
            and not path.name.startswith('~$')
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not ppid_files:
        raise FileNotFoundError(
            'matched_ppid 변수가 없고 '
            'PPID_FILTERED Excel도 없습니다.'
        )

    ppid_excel_path = None
    validated_ufi_targets = None
    validated_reader = None
    ppid_file_errors = []

    # 가장 최근 파일부터 openpyxl -> xlwings 순서로 확인한다.
    for candidate_path in ppid_files:
        try:
            file_size = candidate_path.stat().st_size

            if file_size == 0:
                raise ValueError('0 byte file')

            candidate_data, candidate_reader = read_excel_sheet_robust(
                candidate_path,
                sheet_name='MATCHED_LIST',
            )

            if candidate_data.empty:
                raise ValueError('MATCHED_LIST sheet가 비어 있습니다.')

            ppid_excel_path = candidate_path
            validated_ufi_targets = candidate_data
            validated_reader = candidate_reader
            break

        except Exception as error:
            ppid_file_errors.append({
                'path': str(candidate_path),
                'size_bytes': (
                    candidate_path.stat().st_size
                    if candidate_path.exists()
                    else pd.NA
                ),
                'error': f'{type(error).__name__}: {error}',
            })

            print(
                f'손상되었거나 읽을 수 없는 PPID Excel 건너뜀: '
                f'{candidate_path.name}\n'
                f'  {type(error).__name__}: {error}'
            )

    if ppid_excel_path is None:
        error_details = '\n'.join(
            f'- {record["path"]} / '
            f'{record["size_bytes"]} bytes / '
            f'{record["error"]}'
            for record in ppid_file_errors
        )

        raise ValueError(
            '읽을 수 있는 PPID_FILTERED Excel이 없습니다.\n'
            '2단계를 다시 실행하여 Excel을 생성해야 합니다.\n\n'
            f'{error_details}'
        )

    print(f'PPID 필터 Excel 선택: {ppid_excel_path}')
    print(f'Excel reader: {validated_reader}')

    ufi_targets = validated_ufi_targets

    print(f'PPID 필터 Excel을 사용합니다: {ppid_excel_path}')


if ufi_targets.empty:
    raise ValueError('chip-level UFI 조회 대상 wafer가 없습니다.')

ufi_targets = categorical_to_string(ufi_targets)


# =========================================================
# 4. 대상 column 확인
# =========================================================
ROOT_LOT_COL = find_column(
    ufi_targets,
    ['root_lot_id', 'root_lot_id_ufi']
)

WAFER_COL = find_column(
    ufi_targets,
    ['wafer_id', 'wafer_id_ufi']
)

LINE_COL = find_column(
    ufi_targets,
    [
        'line_id_ufi',
        'source_line_id_ovl',
        'line_id_ovl',
        'line_id',
    ]
)

PROCESS_COL = find_column(
    ufi_targets,
    [
        'process_id_ufi',
        'source_process_id_ovl',
        'process_id_ovl',
        'process_id',
    ]
)

ITEM_COL = find_column(
    ufi_targets,
    ['item_id_ufi', 'item_id']
)

TIME_COL = find_column(
    ufi_targets,
    ['tkout_time_ufi', 'tkout_time']
)

PPID_COL = find_column(
    ufi_targets,
    ['ppid_main_ovl', 'ppid_main']
)

OVL_STEP_COL = find_column(
    ufi_targets,
    [
        'source_step_seq_ovl',
        'step_seq_ovl',
        'step_seq',
    ]
)

print('\n사용 column')
print(f'root lot  : {ROOT_LOT_COL}')
print(f'wafer     : {WAFER_COL}')
print(f'line      : {LINE_COL}')
print(f'process   : {PROCESS_COL}')
print(f'UFI item  : {ITEM_COL}')
print(f'UFI time  : {TIME_COL}')
print(f'OVL PPID  : {PPID_COL}')
print(f'OVL step  : {OVL_STEP_COL}')


# =========================================================
# 5. 전체 조회 기간과 RUN_TAG 설정
# =========================================================
ufi_targets[TIME_COL] = normalize_datetime_series(
    ufi_targets[TIME_COL]
)

if 'UFI_DATE_FROM' in globals():
    QUERY_DATE_FROM = pd.Timestamp(UFI_DATE_FROM).floor('D')
else:
    QUERY_DATE_FROM = (
        ufi_targets[TIME_COL]
        .dropna()
        .min()
        .floor('D')
    )

if 'UFI_DATE_TO_EXCLUSIVE' in globals():
    QUERY_DATE_TO_EXCLUSIVE = pd.Timestamp(
        UFI_DATE_TO_EXCLUSIVE
    ).floor('D')
else:
    QUERY_DATE_TO_EXCLUSIVE = (
        ufi_targets[TIME_COL]
        .dropna()
        .max()
        .floor('D')
        + pd.Timedelta(days=1)
    )

if pd.isna(QUERY_DATE_FROM) or pd.isna(QUERY_DATE_TO_EXCLUSIVE):
    raise ValueError('UFI 조회 기간을 결정할 수 없습니다.')

if QUERY_DATE_FROM >= QUERY_DATE_TO_EXCLUSIVE:
    raise ValueError(
        'UFI 조회 기간이 올바르지 않습니다: '
        f'{QUERY_DATE_FROM} ~ {QUERY_DATE_TO_EXCLUSIVE}'
    )

iso_info = QUERY_DATE_FROM.isocalendar()
AUTO_RUN_TAG = f'{int(iso_info.year) % 100:02d}{int(iso_info.week):02d}'

configured_tag = globals().get('TAG_WEEKNUM')

if configured_tag and str(configured_tag) != AUTO_RUN_TAG:
    warnings.warn(
        f'TAG_WEEKNUM={configured_tag}와 조회 시작일의 ISO 주차 '
        f'{AUTO_RUN_TAG}가 다릅니다. 날짜 기준 {AUTO_RUN_TAG}를 사용합니다.'
    )

RUN_TAG = AUTO_RUN_TAG
TAG_WEEKNUM = RUN_TAG

UFI_DOWNLOAD_DIR = (
    Path(OUTPUT_DIR)
    / 'UFI_CHIP_FILTERED'
    / RUN_TAG
)

UFI_CACHE_DIR = UFI_DOWNLOAD_DIR / 'CACHE'
UFI_QUERY_TEMP_DIR = UFI_DOWNLOAD_DIR / '_QUERY_TEMP'
UFI_PROGRESS_PATH = UFI_DOWNLOAD_DIR / (
    f'UFI_QUERY_PROGRESS__{RUN_TAG}.pkl'
)

UFI_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
UFI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
UFI_QUERY_TEMP_DIR.mkdir(parents=True, exist_ok=True)

print('\nUFI 전체 조회 기간')
print(f'{QUERY_DATE_FROM} 이상')
print(f'{QUERY_DATE_TO_EXCLUSIVE} 미만')
print(f'RUN_TAG: {RUN_TAG}')


# =========================================================
# 6. 대상 wafer/item/측정일 정보 정리
# =========================================================
ufi_targets['_root_key'] = clean_id_series(
    ufi_targets[ROOT_LOT_COL],
    upper=True
)

ufi_targets['_wafer_key'] = clean_id_series(
    ufi_targets[WAFER_COL]
)

ufi_targets['_line_key'] = clean_text_series(
    ufi_targets[LINE_COL]
)

ufi_targets['_process_key'] = clean_text_series(
    ufi_targets[PROCESS_COL]
)

ufi_targets['_target_date'] = (
    ufi_targets[TIME_COL].dt.floor('D')
)

ufi_targets['_item_key'] = ufi_targets[ITEM_COL].apply(
    split_item_value
)

ufi_targets = ufi_targets.explode(
    '_item_key',
    ignore_index=True
)

ufi_targets['_item_key'] = clean_text_series(
    ufi_targets['_item_key']
)

missing_date_rows = int(ufi_targets['_target_date'].isna().sum())

if missing_date_rows:
    raise ValueError(
        f'UFI 측정일이 없는 대상이 {missing_date_rows:,} rows 있습니다. '
        f'{TIME_COL} 값을 확인하세요.'
    )

ufi_target_keys = (
    ufi_targets[
        [
            '_line_key',
            '_process_key',
            '_item_key',
            '_target_date',
            '_root_key',
            '_wafer_key',
        ]
    ]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)

ufi_target_keys = ufi_target_keys[
    ufi_target_keys['_target_date'].ge(QUERY_DATE_FROM)
    & ufi_target_keys['_target_date'].lt(QUERY_DATE_TO_EXCLUSIVE)
].reset_index(drop=True)

if ufi_target_keys.empty:
    raise ValueError('정리 후 남아 있는 UFI 조회 대상이 없습니다.')

target_wafer_count = (
    ufi_target_keys[['_root_key', '_wafer_key']]
    .drop_duplicates()
    .shape[0]
)

target_wafer_item_count = (
    ufi_target_keys[
        ['_item_key', '_root_key', '_wafer_key']
    ]
    .drop_duplicates()
    .shape[0]
)


# =========================================================
# 7. 날짜별 최초 쿼리 계획 생성
# =========================================================
initial_query_plan = []

plan_group_columns = [
    '_line_key',
    '_process_key',
    '_item_key',
    '_target_date',
]

for group_key, group in ufi_target_keys.groupby(
    plan_group_columns,
    sort=False,
    dropna=False,
):
    line_id, process_id, item_id, target_date = group_key

    root_lot_ids = sorted(
        group['_root_key']
        .dropna()
        .unique()
        .tolist()
    )

    for batch_no, root_lot_batch in enumerate(
        split_batches(root_lot_ids, INITIAL_ROOT_LOT_BATCH_SIZE),
        start=1,
    ):
        batch_targets = (
            group[
                group['_root_key'].isin(root_lot_batch)
            ][
                ['_root_key', '_wafer_key']
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        initial_query_plan.append({
            'line_id': line_id,
            'process_id': process_id,
            'item_id': item_id,
            'query_date': pd.Timestamp(target_date),
            'initial_batch_no': batch_no,
            'root_lot_ids': root_lot_batch,
            'target_keys': batch_targets,
            'split_depth': 0,
            'split_path': str(batch_no),
        })

print('\n조회 대상')
print(f'unique wafer       : {target_wafer_count:,}')
print(f'wafer × UFI item   : {target_wafer_item_count:,}')
print(f'최초 root lot batch: {INITIAL_ROOT_LOT_BATCH_SIZE:,}')
print(f'최초 쿼리 계획     : {len(initial_query_plan):,}')
print('timeout 발생 시 해당 배치만 자동으로 절반 분할합니다.')


# =========================================================
# 8. 배치 cache 및 쿼리 함수
# =========================================================
def make_batch_cache_path(plan):
    root_text = '|'.join(sorted(map(str, plan['root_lot_ids'])))

    target_text = '\n'.join(
        plan['target_keys'][['_root_key', '_wafer_key']]
        .astype(str)
        .agg('|'.join, axis=1)
        .sort_values()
        .tolist()
    )

    signature_text = '\n'.join([
        CACHE_VERSION,
        str(plan['line_id']),
        str(plan['process_id']),
        str(plan['item_id']),
        str(pd.Timestamp(plan['query_date']).date()),
        root_text,
        target_text,
        str(USE_WAFER_ID_QUERY_FILTER),
    ])

    signature = hashlib.sha1(
        signature_text.encode('utf-8')
    ).hexdigest()[:14]

    filename = (
        f'DAILY__{safe_token(plan["line_id"])}__'
        f'{safe_token(plan["process_id"])}__'
        f'{safe_token(plan["item_id"])}__'
        f'{pd.Timestamp(plan["query_date"]):%Y%m%d}__'
        f'{len(plan["root_lot_ids"])}LOTS__'
        f'{signature}.pkl'
    )

    return UFI_CACHE_DIR / filename


BDQ_WORKER_PATH = UFI_DOWNLOAD_DIR / '_bdq_query_worker.py'

BDQ_WORKER_CODE = r'''# Auto-generated worker for cancellable bdq query
import pickle
import sys
import traceback

import pandas as pd
import bigdataquery as bdq


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, 'rb') as file:
        params = pickle.load(file)

    data = bdq.getData(param=params)

    if data is None:
        data = pd.DataFrame()

    data.to_pickle(output_path)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
'''


def ensure_bdq_worker():
    current_code = None

    if BDQ_WORKER_PATH.exists():
        try:
            current_code = BDQ_WORKER_PATH.read_text(encoding='utf-8')
        except Exception:
            current_code = None

    if current_code != BDQ_WORKER_CODE:
        BDQ_WORKER_PATH.write_text(
            BDQ_WORKER_CODE,
            encoding='utf-8'
        )


def run_bdq_in_subprocess(params):
    """
    bdq.getData를 별도 process에서 실행한다.

    Thread timeout과 달리 hard timeout 시 worker process 자체를 종료하므로
    ODBC가 몇 시간 동안 반환되지 않아 notebook 전체가 멈추는 것을 막는다.
    """
    ensure_bdq_worker()

    task_id = uuid.uuid4().hex
    input_path = UFI_QUERY_TEMP_DIR / f'{task_id}__input.pkl'
    output_path = UFI_QUERY_TEMP_DIR / f'{task_id}__output.pkl'
    worker_log_path = UFI_QUERY_TEMP_DIR / f'{task_id}__worker.log'

    with input_path.open('wb') as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

    process = None
    worker_log_handle = None
    timed_out = False
    worker_log_text = ''

    try:
        worker_log_handle = worker_log_path.open(
            'w',
            encoding='utf-8',
            errors='replace',
        )

        process = subprocess.Popen(
            [
                sys.executable,
                str(BDQ_WORKER_PATH),
                str(input_path),
                str(output_path),
            ],
            stdout=worker_log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
        )

        start_time = time.time()
        next_heartbeat = QUERY_HEARTBEAT_SEC

        while process.poll() is None:
            elapsed = time.time() - start_time

            if elapsed >= CLIENT_HARD_TIMEOUT_SEC:
                timed_out = True

                try:
                    process.terminate()
                    process.wait(timeout=10)
                except Exception:
                    try:
                        process.kill()
                        process.wait(timeout=10)
                    except Exception:
                        pass

                break

            if elapsed >= next_heartbeat:
                print(
                    f'  ... bdq query 실행 중: '
                    f'{elapsed / 60:,.1f}분'
                )
                next_heartbeat += QUERY_HEARTBEAT_SEC

            time.sleep(5)

        if worker_log_handle is not None:
            worker_log_handle.close()
            worker_log_handle = None

        if worker_log_path.exists():
            worker_log_text = worker_log_path.read_text(
                encoding='utf-8',
                errors='replace',
            )

        if timed_out:
            raise TimeoutError(
                f'client hard timeout after '
                f'{CLIENT_HARD_TIMEOUT_SEC / 60:.0f} minutes. '
                'Worker process was terminated.'
            )

        if process.returncode != 0:
            log_tail = worker_log_text[-5000:]
            raise RuntimeError(
                'bdq worker query failed.\n'
                f'{log_tail}'
            )

        if not output_path.exists():
            raise RuntimeError(
                'bdq worker가 정상 종료되었지만 결과 파일이 없습니다.'
            )

        return pd.read_pickle(output_path)

    finally:
        if worker_log_handle is not None:
            try:
                worker_log_handle.close()
            except Exception:
                pass

        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        for temp_path in [input_path, output_path, worker_log_path]:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def query_chip_ufi_batch(plan, use_wafer_filter):
    query_date_from = pd.Timestamp(plan['query_date']).floor('D')
    query_date_to = query_date_from + pd.Timedelta(days=1)

    params = {
        'table_name': 'eds.m_eds_chip_measure',
        'dateFrom': query_date_from.strftime('%Y-%m-%d'),
        'dateTo': query_date_to.strftime('%Y-%m-%d'),
        'step_seq': 'EDS_P',
        'line_id': plan['line_id'],
        'process_id': plan['process_id'],
        'item_type': ['UFI'],
        'item_id': [plan['item_id']],
        'subitem_id': UFI_SUBITEM_ID,
        'root_lot_id': list(plan['root_lot_ids']),
        'setLimit': False,
    }

    if use_wafer_filter:
        params['wafer_id'] = sorted(
            plan['target_keys']['_wafer_key']
            .dropna()
            .unique()
            .tolist()
        )

    return run_bdq_in_subprocess(params)


def split_plan_in_half(plan):
    root_lot_ids = list(plan['root_lot_ids'])

    if len(root_lot_ids) <= MIN_ROOT_LOT_BATCH_SIZE:
        return []

    midpoint = max(1, len(root_lot_ids) // 2)
    root_halves = [
        root_lot_ids[:midpoint],
        root_lot_ids[midpoint:],
    ]

    child_plans = []

    for child_no, child_roots in enumerate(root_halves, start=1):
        if not child_roots:
            continue

        child_targets = (
            plan['target_keys'][
                plan['target_keys']['_root_key'].isin(child_roots)
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        child_plan = dict(plan)
        child_plan['root_lot_ids'] = child_roots
        child_plan['target_keys'] = child_targets
        child_plan['split_depth'] = plan['split_depth'] + 1
        child_plan['split_path'] = (
            f'{plan["split_path"]}.{child_no}'
        )

        child_plans.append(child_plan)

    return child_plans


# =========================================================
# 9. 날짜/배치 단위 조회 및 timeout 자동 분할
# =========================================================
ufi_frames = []
ufi_query_logs = []

run_counter = 0


def append_query_log(
    plan,
    status,
    requested_wafers,
    found_wafers=0,
    query_rows=0,
    filtered_rows=0,
    elapsed_sec=0,
    cache_path='',
    wafer_filter_used=pd.NA,
    error='',
):
    ufi_query_logs.append({
        'run_no': run_counter,
        'line_id': plan['line_id'],
        'process_id': plan['process_id'],
        'item_id': plan['item_id'],
        'query_date': plan['query_date'],
        'initial_batch_no': plan['initial_batch_no'],
        'split_depth': plan['split_depth'],
        'split_path': plan['split_path'],
        'root_lot_count': len(plan['root_lot_ids']),
        'root_lot_ids': '|'.join(plan['root_lot_ids']),
        'status': status,
        'requested_wafers': requested_wafers,
        'found_wafers': found_wafers,
        'query_rows': query_rows,
        'filtered_rows': filtered_rows,
        'elapsed_sec': elapsed_sec,
        'wafer_filter_used': wafer_filter_used,
        'cache_path': str(cache_path),
        'error': str(error),
    })

    # Notebook/PC가 중단되어도 현재까지의 진행상태를 확인할 수 있게
    # 매 단계마다 가벼운 Pickle checkpoint를 갱신한다.
    try:
        pd.DataFrame(ufi_query_logs).to_pickle(UFI_PROGRESS_PATH)
    except Exception as checkpoint_error:
        print(f'진행 로그 checkpoint 저장 실패: {checkpoint_error}')


def process_query_plan(plan, initial_no, initial_total):
    global run_counter

    cache_path = make_batch_cache_path(plan)
    split_marker_path = cache_path.with_name(
        f'{cache_path.stem}__SPLIT.pkl'
    )

    requested_wafers = len(plan['target_keys'])

    # 이전 실행에서 timeout으로 분할된 부모 배치는 다시 조회하지 않는다.
    if split_marker_path.exists():
        print('\n' + '-' * 80)
        print(
            f'[PLAN {initial_no}/{initial_total}] '
            f'{plan["line_id"]} / {plan["process_id"]} / '
            f'{plan["item_id"]} / {plan["query_date"]:%Y-%m-%d}'
        )
        print(
            f'split marker 사용: {len(plan["root_lot_ids"])} LOT '
            f'→ 하위 배치'
        )

        append_query_log(
            plan=plan,
            status='SPLIT_CACHE',
            requested_wafers=requested_wafers,
            cache_path=split_marker_path,
        )

        for child_plan in split_plan_in_half(plan):
            process_query_plan(child_plan, initial_no, initial_total)

        return

    # 완료된 cache 사용
    if cache_path.exists():
        try:
            chip_filtered = pd.read_pickle(cache_path)

            # 이전 버전에서 subitem 조건 없이 저장한 cache도 재사용하되,
            # 로딩 시 A_CSUM_PERCENT만 남겨 동일한 기준으로 변환한다.
            cached_rows_before = len(chip_filtered)
            chip_filtered = filter_ufi_subitem(chip_filtered)

            if (
                SAVE_QUERY_CACHE
                and len(chip_filtered) != cached_rows_before
            ):
                chip_filtered.to_pickle(cache_path)
                print(
                    f'기존 cache subitem 정리: '
                    f'{cached_rows_before:,} → {len(chip_filtered):,} rows'
                )

            if not chip_filtered.empty:
                ufi_frames.append(chip_filtered)

            found_wafers = found_wafer_count(chip_filtered)

            print('\n' + '-' * 80)
            print(
                f'[PLAN {initial_no}/{initial_total}] cache / '
                f'{plan["process_id"]} / {plan["item_id"]} / '
                f'{plan["query_date"]:%Y-%m-%d} / '
                f'{len(plan["root_lot_ids"])} LOT'
            )
            print(
                f'cache rows={len(chip_filtered):,}, '
                f'wafer={found_wafers:,}/{requested_wafers:,}'
            )

            append_query_log(
                plan=plan,
                status='CACHE',
                requested_wafers=requested_wafers,
                found_wafers=found_wafers,
                filtered_rows=len(chip_filtered),
                cache_path=cache_path,
            )

            return

        except Exception as error:
            print(f'cache 읽기 실패 → 다시 조회: {error}')

    run_counter += 1

    print('\n' + '=' * 80)
    print(
        f'[PLAN {initial_no}/{initial_total} | RUN {run_counter}] '
        f'{plan["line_id"]} / {plan["process_id"]} / '
        f'{plan["item_id"]} / {plan["query_date"]:%Y-%m-%d}'
    )
    print(
        f'배치 경로: {plan["split_path"]} / '
        f'split depth: {plan["split_depth"]}'
    )
    print(f'root lot: {len(plan["root_lot_ids"]):,}')
    print(f'요청 wafer: {requested_wafers:,}')

    query_start = time.time()
    query_succeeded = False
    data_chip = None
    last_error = None
    split_required = False
    wafer_filter_used = False

    filter_modes = (
        [True, False]
        if USE_WAFER_ID_QUERY_FILTER
        else [False]
    )

    for use_wafer_filter in filter_modes:
        # wafer filter 사용 시 지원 여부 확인용으로 1회 시도한다.
        # filter 없이 실행할 때만 일반 오류 재시도를 적용한다.
        retry_count = 1 if use_wafer_filter else QUERY_RETRY

        for attempt in range(1, retry_count + 1):
            try:
                data_chip = query_chip_ufi_batch(
                    plan=plan,
                    use_wafer_filter=use_wafer_filter,
                )

                query_succeeded = True
                wafer_filter_used = use_wafer_filter
                last_error = None
                break

            except Exception as error:
                last_error = error

                print(
                    f'쿼리 실패: {type(error).__name__}: {error}'
                )

                if is_timeout_or_resource_error(error):
                    split_required = True
                    break

                if use_wafer_filter:
                    print(
                        'wafer_id DB 조건 없이 다시 시도합니다.'
                    )
                    break

                if attempt < retry_count:
                    print(
                        f'일시적 오류 재시도 '
                        f'{attempt + 1}/{retry_count}'
                    )
                    time.sleep(QUERY_RETRY_WAIT_SEC)

        if query_succeeded or split_required:
            break

    elapsed_sec = round(time.time() - query_start, 1)

    # -----------------------------------------------------
    # timeout/resource 오류: 해당 root lot 배치만 절반 분할
    # -----------------------------------------------------
    if not query_succeeded:
        child_plans = (
            split_plan_in_half(plan)
            if split_required
            else []
        )

        if child_plans:
            pd.to_pickle(
                {
                    'status': 'SPLIT',
                    'created_at': datetime.now(),
                    'reason': str(last_error),
                    'root_lot_ids': plan['root_lot_ids'],
                },
                split_marker_path,
            )

            append_query_log(
                plan=plan,
                status='SPLIT_TIMEOUT',
                requested_wafers=requested_wafers,
                elapsed_sec=elapsed_sec,
                cache_path=split_marker_path,
                wafer_filter_used=wafer_filter_used,
                error=last_error,
            )

            print(
                f'timeout 배치 자동 분할: '
                f'{len(plan["root_lot_ids"])} LOT → '
                f'{[len(child["root_lot_ids"]) for child in child_plans]}'
            )

            for child_plan in child_plans:
                process_query_plan(
                    child_plan,
                    initial_no,
                    initial_total,
                )

            return

        append_query_log(
            plan=plan,
            status=(
                'FAILED_TIMEOUT'
                if split_required
                else 'FAILED'
            ),
            requested_wafers=requested_wafers,
            elapsed_sec=elapsed_sec,
            wafer_filter_used=wafer_filter_used,
            error=last_error,
        )

        print('더 이상 분할할 수 없어 실패 처리합니다.')
        return

    # -----------------------------------------------------
    # 쿼리 성공 후 정확한 대상 wafer만 유지
    # -----------------------------------------------------
    if data_chip is None:
        data_chip = pd.DataFrame()

    data_chip = categorical_to_string(data_chip)
    data_chip = filter_ufi_subitem(data_chip)
    query_rows = len(data_chip)

    if data_chip.empty:
        chip_filtered = pd.DataFrame()

        if SAVE_QUERY_CACHE:
            chip_filtered.to_pickle(cache_path)

        append_query_log(
            plan=plan,
            status='EMPTY',
            requested_wafers=requested_wafers,
            query_rows=0,
            filtered_rows=0,
            elapsed_sec=elapsed_sec,
            cache_path=(cache_path if SAVE_QUERY_CACHE else ''),
            wafer_filter_used=wafer_filter_used,
        )

        print('조회 결과가 없습니다.')
        return

    chip_root_lot_col = find_column(
        data_chip,
        ['root_lot_id', 'root_lot_id_ufi']
    )

    chip_wafer_col = find_column(
        data_chip,
        ['wafer_id', 'wafer_id_ufi']
    )

    chip_item_col = find_column(
        data_chip,
        ['item_id'],
        required=False
    )

    data_chip['_root_key'] = clean_id_series(
        data_chip[chip_root_lot_col],
        upper=True
    )

    data_chip['_wafer_key'] = clean_id_series(
        data_chip[chip_wafer_col]
    )

    if chip_item_col is not None:
        data_chip['_returned_item_key'] = clean_text_series(
            data_chip[chip_item_col]
        )

        data_chip = data_chip[
            data_chip['_returned_item_key'].eq(plan['item_id'])
        ].copy()

    chip_filtered = pd.merge(
        data_chip,
        plan['target_keys'].drop_duplicates(),
        on=['_root_key', '_wafer_key'],
        how='inner',
        validate='many_to_one',
    )

    query_date_from = pd.Timestamp(plan['query_date']).floor('D')
    query_date_to = query_date_from + pd.Timedelta(days=1)

    if 'tkout_time' in chip_filtered.columns:
        chip_filtered['tkout_time'] = normalize_datetime_series(
            chip_filtered['tkout_time']
        )

        chip_filtered = chip_filtered[
            chip_filtered['tkout_time'].ge(query_date_from)
            & chip_filtered['tkout_time'].lt(query_date_to)
        ].copy()

    chip_filtered['root_lot_id_key'] = chip_filtered['_root_key']
    chip_filtered['wafer_id_key'] = chip_filtered['_wafer_key']

    chip_filtered['rlid_wid'] = (
        chip_filtered['root_lot_id_key']
        + '_'
        + chip_filtered['wafer_id_key']
    )

    chip_filtered['query_line_id'] = plan['line_id']
    chip_filtered['query_process_id'] = plan['process_id']
    chip_filtered['query_item_id'] = plan['item_id']
    chip_filtered['source_query_date'] = query_date_from
    chip_filtered['query_initial_batch_no'] = plan['initial_batch_no']
    chip_filtered['query_split_depth'] = plan['split_depth']
    chip_filtered['query_split_path'] = plan['split_path']
    chip_filtered['query_root_lot_count'] = len(plan['root_lot_ids'])
    chip_filtered['query_wafer_filter_used'] = wafer_filter_used
    chip_filtered['source_query_time'] = datetime.now()

    chip_filtered = chip_filtered.drop(
        columns=[
            '_root_key',
            '_wafer_key',
            '_returned_item_key',
        ],
        errors='ignore',
    )

    found_wafers = found_wafer_count(chip_filtered)

    if SAVE_QUERY_CACHE:
        chip_filtered.to_pickle(cache_path)

    if not chip_filtered.empty:
        ufi_frames.append(chip_filtered)

    append_query_log(
        plan=plan,
        status='QUERIED',
        requested_wafers=requested_wafers,
        found_wafers=found_wafers,
        query_rows=query_rows,
        filtered_rows=len(chip_filtered),
        elapsed_sec=elapsed_sec,
        cache_path=(cache_path if SAVE_QUERY_CACHE else ''),
        wafer_filter_used=wafer_filter_used,
    )

    print(f'쿼리 row: {query_rows:,}')
    print(f'필터 row: {len(chip_filtered):,}')
    print(f'확인 wafer: {found_wafers:,}/{requested_wafers:,}')
    print(f'소요 시간: {elapsed_sec:,.1f}초')

    if QUERY_PAUSE_SEC > 0:
        time.sleep(QUERY_PAUSE_SEC)


for initial_no, plan in enumerate(initial_query_plan, start=1):
    process_query_plan(
        plan=plan,
        initial_no=initial_no,
        initial_total=len(initial_query_plan),
    )


# =========================================================
# 10. 조회 결과 통합 및 target coverage 확인
# =========================================================
df_ufi_query_log = pd.DataFrame(ufi_query_logs)

if not ufi_frames:
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    log_path = UFI_DOWNLOAD_DIR / (
        f'UFI_QUERY_FAILED_LOG__{timestamp}.xlsx'
    )

    df_ufi_query_log.to_excel(log_path, index=False)

    raise ValueError(
        '조회된 chip-level UFI 데이터가 없습니다.\n'
        f'로그: {log_path}'
    )

ufi_chip_filtered_all = pd.concat(
    ufi_frames,
    ignore_index=True
)

ufi_chip_filtered_all = (
    ufi_chip_filtered_all
    .drop_duplicates()
    .reset_index(drop=True)
)

downloaded_target_keys = (
    ufi_chip_filtered_all[
        [
            'query_line_id',
            'query_process_id',
            'query_item_id',
            'source_query_date',
            'root_lot_id_key',
            'wafer_id_key',
        ]
    ]
    .rename(columns={
        'query_line_id': '_line_key',
        'query_process_id': '_process_key',
        'query_item_id': '_item_key',
        'source_query_date': '_target_date',
        'root_lot_id_key': '_root_key',
        'wafer_id_key': '_wafer_key',
    })
    .drop_duplicates()
)

downloaded_target_keys['_target_date'] = normalize_datetime_series(
    downloaded_target_keys['_target_date']
).dt.floor('D')

coverage_keys = [
    '_line_key',
    '_process_key',
    '_item_key',
    '_target_date',
    '_root_key',
    '_wafer_key',
]

df_ufi_target_coverage = pd.merge(
    ufi_target_keys,
    downloaded_target_keys.assign(ufi_data_found=True),
    on=coverage_keys,
    how='left',
    validate='one_to_one',
)

df_ufi_target_coverage['ufi_data_found'] = (
    df_ufi_target_coverage['ufi_data_found']
    .fillna(False)
    .astype(bool)
)

df_ufi_missing_targets = df_ufi_target_coverage[
    ~df_ufi_target_coverage['ufi_data_found']
].copy()


# =========================================================
# 11. 통합 Pickle과 쿼리 로그 저장
# =========================================================
timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

UFI_PICKLE_PATH = UFI_DOWNLOAD_DIR / (
    f'UFI_CHIP_FILTERED_ALL__{timestamp}.pkl'
)

UFI_LOG_PATH = UFI_DOWNLOAD_DIR / (
    f'UFI_QUERY_LOG__{timestamp}.xlsx'
)

if SAVE_COMBINED_PICKLE:
    ufi_chip_filtered_all.to_pickle(UFI_PICKLE_PATH)

with pd.ExcelWriter(UFI_LOG_PATH, engine='openpyxl') as writer:
    df_ufi_query_log.to_excel(
        writer,
        sheet_name='QUERY_LOG',
        index=False
    )

    df_ufi_target_coverage.to_excel(
        writer,
        sheet_name='TARGET_COVERAGE',
        index=False
    )

    df_ufi_missing_targets.to_excel(
        writer,
        sheet_name='MISSING_TARGETS',
        index=False
    )


# =========================================================
# 12. 결과 확인
# =========================================================
downloaded_wafer_count = (
    ufi_chip_filtered_all[
        ['root_lot_id_key', 'wafer_id_key']
    ]
    .drop_duplicates()
    .shape[0]
)

downloaded_wafer_item_count = (
    downloaded_target_keys[
        ['_item_key', '_root_key', '_wafer_key']
    ]
    .drop_duplicates()
    .shape[0]
)

if 'item_id' in ufi_chip_filtered_all.columns:
    found_item_ids = sorted(
        clean_text_series(ufi_chip_filtered_all['item_id'])
        .dropna()
        .unique()
        .tolist()
    )
else:
    found_item_ids = sorted(
        clean_text_series(ufi_chip_filtered_all['query_item_id'])
        .dropna()
        .unique()
        .tolist()
    )

failed_query_count = int(
    df_ufi_query_log['status']
    .isin(['FAILED', 'FAILED_TIMEOUT'])
    .sum()
)

split_query_count = int(
    df_ufi_query_log['status']
    .eq('SPLIT_TIMEOUT')
    .sum()
)

print('\n' + '=' * 80)
print(f'최초 쿼리 계획: {len(initial_query_plan):,}')
print(f'실제 DB 쿼리 실행: {run_counter:,}')
print(f'timeout 자동 분할: {split_query_count:,}')
print(f'최종 실패 배치: {failed_query_count:,}')
print(f'요청 unique wafer: {target_wafer_count:,}')
print(f'UFI 확인 unique wafer: {downloaded_wafer_count:,}')
print(f'요청 wafer × item: {target_wafer_item_count:,}')
print(f'확인 wafer × item: {downloaded_wafer_item_count:,}')
print(f'누락 wafer × item × date: {len(df_ufi_missing_targets):,}')
print(f'chip-level UFI row: {len(ufi_chip_filtered_all):,}')
print(f'UFI item: {found_item_ids}')

if SAVE_COMBINED_PICKLE:
    print(f'UFI Pickle: {UFI_PICKLE_PATH}')

print(f'쿼리/coverage 로그: {UFI_LOG_PATH}')
print(f'실행 중 progress checkpoint: {UFI_PROGRESS_PATH}')

if failed_query_count > 0:
    warnings.warn(
        f'최종 실패 배치가 {failed_query_count:,}개 있습니다. '
        'QUERY_LOG를 확인하고 다시 실행하세요. '
        '완료 배치와 split 정보는 cache를 사용합니다.'
    )

display(df_ufi_query_log)
