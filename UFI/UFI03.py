# 3단계 수정본
"""
CODE VERSION: STAGE3_V2_CONTEXT_PRESERVED_20260901

3단계 수정 내용
---------------
1) 2단계 PPID_FILTERED Excel의 MATCHED_LIST에서 아래 context를 읽는다.
   - eqp_id_main_ovl
   - ppid_main_ovl
   - reticle_id_main_ovl
   - eqp_id_base_ovl
   - ppid_base_ovl
   - reticle_id_base_ovl
2) MOD_DOWNLOAD_TARGETS와 wafer/process/step/PPID/mod path 기준으로 결합한다.
3) FTP _mod_ 데이터의 chip 대상 행에 context를 추가한다.
4) 파일별 cache와 OVL_FILTERED_ALL Pickle에 context를 보존한다.

주의
----
- 이전 3단계 cache에는 context가 없으므로 CACHE_VERSION을 바꿨다.
  _mod_ 파일은 최초 1회 다시 다운로드한다.
- 4단계 UFI query cache에는 영향을 주지 않는다.
"""

from ftplib import FTP
from io import BytesIO
from pathlib import Path
from datetime import datetime

import hashlib
import posixpath
import time

import numpy as np
import pandas as pd


CODE_VERSION = 'STAGE3_V2_CONTEXT_PRESERVED_V0901'
CACHE_VERSION = 'OVL_MOD_CONTEXT_CACHE_V2'

print(f'[CODE VERSION] {CODE_VERSION}')


# =========================================================
# 1. 사용자 설정
# =========================================================
if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = Path.cwd() / 'UFI_OVL_OUTPUT'
else:
    OUTPUT_DIR = Path(OUTPUT_DIR)

OVL_DOWNLOAD_DIR = OUTPUT_DIR / 'OVL_MOD_FILTERED'
OVL_CACHE_DIR = OVL_DOWNLOAD_DIR / 'CACHE'

OVL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OVL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

FTP_RETRY = 3
SAVE_COMBINED_CSV = False


CONTEXT_COLUMNS = [
    'eqp_id_main_ovl',
    'ppid_main_ovl',
    'reticle_id_main_ovl',
    'eqp_id_base_ovl',
    'ppid_base_ovl',
    'reticle_id_base_ovl',
]


# 특정 2단계 Excel을 직접 지정할 때만 Path(...) 입력
PPID_FILTER_EXCEL_PATH = None


# =========================================================
# 2. 공통 함수
# =========================================================
def find_column_ci(df, candidates, required=True):
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        found = lookup.get(str(candidate).strip().lower())

        if found is not None:
            return found

    if required:
        raise KeyError(
            '다음 column 중 사용할 수 있는 column이 없습니다.\n'
            f'후보: {candidates}\n\n'
            f'현재 column:\n{df.columns.tolist()}'
        )

    return None


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
    return series.astype('string').str.strip().str.upper()


def clean_remote_path_series(series):
    return (
        series
        .astype('string')
        .str.strip()
        .str.replace('\\', '/', regex=False)
        .str.replace('_list_', '_mod_', regex=False)
    )


def unique_text_values(series):
    values = clean_text_series(series).dropna()
    return sorted({
        str(value).strip().upper()
        for value in values
        if str(value).strip()
    })


def make_unique_headers(headers):
    result = []
    counts = {}

    for index, header in enumerate(headers, start=1):
        base = str(header).strip() if header is not None else ''
        base = base or f'COLUMN_{index}'
        count = counts.get(base, 0)
        counts[base] = count + 1
        result.append(base if count == 0 else f'{base}__DUP{count}')

    return result


def read_excel_sheet_robust(path, sheet_name):
    """openpyxl 실패 시 사내 DRM 대응을 위해 xlwings로 다시 읽는다."""
    path = Path(path)
    openpyxl_error = None

    try:
        result = pd.read_excel(
            path,
            sheet_name=sheet_name,
            dtype=str,
            engine='openpyxl',
        )
        print(f'openpyxl로 읽음: {path.name} / {sheet_name}')
        return result

    except Exception as error:
        openpyxl_error = error
        print(
            f'openpyxl 읽기 실패 -> Excel COM 시도: '
            f'{path.name} / {type(error).__name__}: {error}'
        )

    try:
        import xlwings as xw

        app = None
        book = None

        try:
            app = xw.App(visible=False, add_book=False)
            app.display_alerts = False
            app.screen_updating = False

            book = app.books.open(
                str(path.resolve()),
                update_links=False,
                read_only=True,
            )

            sheet_names = [sheet.name for sheet in book.sheets]

            if sheet_name not in sheet_names:
                raise KeyError(
                    f'{sheet_name} sheet가 없습니다. 현재 sheet: {sheet_names}'
                )

            values = book.sheets[sheet_name].used_range.value

            if values is None:
                return pd.DataFrame()

            if not isinstance(values, list):
                values = [[values]]
            elif values and not isinstance(values[0], list):
                values = [values]

            if len(values) < 2:
                return pd.DataFrame(
                    columns=make_unique_headers(values[0] if values else [])
                )

            headers = make_unique_headers(values[0])
            result = pd.DataFrame(values[1:], columns=headers)
            result = result.dropna(how='all').reset_index(drop=True)

            print(f'Excel COM(xlwings)으로 읽음: {path.name} / {sheet_name}')
            return result

        finally:
            if book is not None:
                try:
                    book.close()
                except Exception:
                    pass

            if app is not None:
                try:
                    app.quit()
                except Exception:
                    pass

    except Exception as xlwings_error:
        raise ValueError(
            f'Excel을 읽지 못했습니다: {path}\n'
            f'openpyxl: {type(openpyxl_error).__name__}: {openpyxl_error}\n'
            f'xlwings: {type(xlwings_error).__name__}: {xlwings_error}'
        ) from xlwings_error


def find_latest_readable_ppid_excel():
    if PPID_FILTER_EXCEL_PATH is not None:
        candidates = [Path(PPID_FILTER_EXCEL_PATH)]
    else:
        candidates = sorted(
            [
                path
                for path in OUTPUT_DIR.glob('PPID_FILTERED__*.xlsx')
                if path.is_file()
                and not path.name.startswith('~$')
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    if not candidates:
        raise FileNotFoundError(
            f'PPID_FILTERED Excel이 없습니다: {OUTPUT_DIR}'
        )

    errors = []

    for path in candidates:
        try:
            matched = read_excel_sheet_robust(path, 'MATCHED_LIST')
            targets = read_excel_sheet_robust(path, 'MOD_DOWNLOAD_TARGETS')

            if matched.empty or targets.empty:
                raise ValueError(
                    'MATCHED_LIST 또는 MOD_DOWNLOAD_TARGETS가 비어 있습니다.'
                )

            return path, matched, targets

        except Exception as error:
            errors.append(
                f'{path.name}: {type(error).__name__}: {error}'
            )
            print(f'읽을 수 없는 PPID Excel 건너뜀: {path.name}')

    raise ValueError(
        '읽을 수 있는 PPID_FILTERED Excel이 없습니다.\n\n'
        + '\n'.join(errors)
    )


def connect_ftp_stage3():
    required_variables = [
        'ftp_host',
        'ftp_port',
        'ftp_user',
        'ftp_password',
    ]
    missing_variables = [
        name for name in required_variables if name not in globals()
    ]

    if missing_variables:
        raise NameError(
            f'FTP 접속 정보가 정의되지 않았습니다: {missing_variables}\n'
            '1단계 코드를 먼저 실행해주세요.'
        )

    timeout = globals().get('FTP_TIMEOUT', 120)
    passive_mode = globals().get('FTP_PASSIVE_MODE', True)

    ftp = FTP()
    ftp.connect(ftp_host, ftp_port, timeout=timeout)
    ftp.login(user=ftp_user, passwd=ftp_password)
    ftp.set_pasv(passive_mode)
    ftp.encoding = 'utf-8'

    return ftp


def close_ftp_stage3(ftp):
    if ftp is None:
        return

    try:
        ftp.quit()
    except Exception:
        try:
            ftp.close()
        except Exception:
            pass


def read_csv_bytes(raw_bytes):
    last_error = None

    for encoding in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(
                BytesIO(raw_bytes),
                encoding=encoding,
                on_bad_lines='skip',
                low_memory=False,
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


def download_csv_from_ftp(ftp, remote_path):
    buffer = BytesIO()
    ftp.retrbinary(f'RETR {remote_path}', buffer.write)
    return read_csv_bytes(buffer.getvalue())


def collapse_context_group(group):
    output = {}
    ambiguous_columns = []
    present_count = 0

    for column in CONTEXT_COLUMNS:
        values = unique_text_values(group[column])

        if not values:
            output[column] = pd.NA
            continue

        present_count += 1
        output[column] = '|'.join(values)

        if len(values) > 1:
            ambiguous_columns.append(column)

    if ambiguous_columns:
        status = 'AMBIGUOUS'
    elif present_count == len(CONTEXT_COLUMNS):
        status = 'MATCHED'
    elif present_count > 0:
        status = 'PARTIAL'
    else:
        status = 'NO_MAPPING'

    output['context_mapping_status'] = status
    output['context_ambiguous_columns'] = '|'.join(ambiguous_columns)

    return pd.Series(output)


def aggregate_target_meta(group, process_col, step_col, ppid_col):
    output = {
        'selected_process_id': (
            '|'.join(unique_text_values(group[process_col]))
        ),
        'selected_step_seq': (
            '|'.join(unique_text_values(group[step_col]))
        ),
        'selected_ppid_main_ovl': (
            '|'.join(unique_text_values(group[ppid_col]))
        ),
    }

    context_result = collapse_context_group(group).to_dict()
    output.update(context_result)

    upstream_status = set(
        clean_text_series(group['context_mapping_status'])
        .dropna()
        .astype(str)
        .tolist()
    )

    if 'AMBIGUOUS' in upstream_status:
        output['context_mapping_status'] = 'AMBIGUOUS'
    elif (
        output['context_mapping_status'] == 'MATCHED'
        and 'PARTIAL' in upstream_status
    ):
        output['context_mapping_status'] = 'PARTIAL'

    return pd.Series(output)


def make_cache_path(remote_path, target_group, root_col, wafer_col):
    signature_columns = [
        root_col,
        wafer_col,
        *CONTEXT_COLUMNS,
        'context_mapping_status',
    ]
    signature_columns = [
        column for column in signature_columns if column in target_group.columns
    ]

    signature_rows = (
        target_group[signature_columns]
        .astype('string')
        .fillna('<NA>')
        .agg('|'.join, axis=1)
        .sort_values()
        .tolist()
    )

    signature_text = '\n'.join([
        CACHE_VERSION,
        str(remote_path),
        *signature_rows,
    ])

    signature = hashlib.sha1(
        signature_text.encode('utf-8')
    ).hexdigest()[:12]

    remote_stem = Path(posixpath.basename(remote_path)).stem[:80]

    return OVL_CACHE_DIR / (
        f'{remote_stem}__{signature}__context_filtered.pkl'
    )


# =========================================================
# 3. 2단계 다운로드 대상과 MATCHED_LIST 가져오기
# =========================================================
ppid_excel_path = None
excel_matched = None
excel_targets = None

need_excel = not (
    'matched_ppid' in globals()
    and isinstance(globals()['matched_ppid'], pd.DataFrame)
    and not globals()['matched_ppid'].empty
    and 'mod_download_targets_ppid' in globals()
    and isinstance(globals()['mod_download_targets_ppid'], pd.DataFrame)
    and not globals()['mod_download_targets_ppid'].empty
)

if need_excel:
    ppid_excel_path, excel_matched, excel_targets = (
        find_latest_readable_ppid_excel()
    )
    print(f'2단계 PPID Excel 사용: {ppid_excel_path}')


if (
    'mod_download_targets_ppid' in globals()
    and isinstance(globals()['mod_download_targets_ppid'], pd.DataFrame)
    and not globals()['mod_download_targets_ppid'].empty
):
    ovl_targets = globals()['mod_download_targets_ppid'].copy()
    print('현재 메모리의 mod_download_targets_ppid를 사용합니다.')
else:
    ovl_targets = excel_targets.copy()
    print('PPID Excel의 MOD_DOWNLOAD_TARGETS를 사용합니다.')


if (
    'matched_ppid' in globals()
    and isinstance(globals()['matched_ppid'], pd.DataFrame)
    and not globals()['matched_ppid'].empty
):
    matched_context_source = globals()['matched_ppid'].copy()
    print('현재 메모리의 matched_ppid에서 context를 가져옵니다.')
else:
    matched_context_source = excel_matched.copy()
    print('PPID Excel의 MATCHED_LIST에서 context를 가져옵니다.')


if ovl_targets.empty:
    raise ValueError('OVL 다운로드 대상 wafer가 없습니다.')

if matched_context_source.empty:
    raise ValueError('context를 가져올 MATCHED_LIST가 비어 있습니다.')


# =========================================================
# 4. 대상 및 context source column 확인
# =========================================================
TARGET_ROOT_COL = find_column_ci(
    ovl_targets,
    ['root_lot_id', 'root_lot_id_ovl'],
)
TARGET_WAFER_COL = find_column_ci(
    ovl_targets,
    ['wafer_id', 'wafer_id_ovl'],
)
TARGET_MOD_PATH_COL = find_column_ci(
    ovl_targets,
    ['ovl_mod_path_ovl', 'ovl_mod_path'],
)
TARGET_PROCESS_COL = find_column_ci(
    ovl_targets,
    [
        'source_process_id_ovl',
        'process_id_ovl',
        'target_process_id',
        'process_id',
    ],
)
TARGET_STEP_COL = find_column_ci(
    ovl_targets,
    [
        'source_step_seq_ovl',
        'step_seq_ovl',
        'target_step_seq',
        'step_seq',
    ],
)
TARGET_PPID_COL = find_column_ci(
    ovl_targets,
    ['ppid_main_ovl', 'ppid_main'],
)


SOURCE_ROOT_COL = find_column_ci(
    matched_context_source,
    ['root_lot_id', 'root_lot_id_ovl'],
)
SOURCE_WAFER_COL = find_column_ci(
    matched_context_source,
    ['wafer_id', 'wafer_id_ovl'],
)
SOURCE_MOD_PATH_COL = find_column_ci(
    matched_context_source,
    ['ovl_mod_path_ovl', 'ovl_mod_path'],
)
SOURCE_PROCESS_COL = find_column_ci(
    matched_context_source,
    [
        'source_process_id_ovl',
        'process_id_ovl',
        'target_process_id',
        'process_id',
    ],
)
SOURCE_STEP_COL = find_column_ci(
    matched_context_source,
    [
        'source_step_seq_ovl',
        'step_seq_ovl',
        'target_step_seq',
        'step_seq',
    ],
)

SOURCE_CONTEXT_COLUMN_MAP = {
    column: find_column_ci(matched_context_source, [column])
    for column in CONTEXT_COLUMNS
}


print('\n보존할 context column')
for output_column, source_column in SOURCE_CONTEXT_COLUMN_MAP.items():
    print(f'{output_column}: {source_column}')


# =========================================================
# 5. MATCHED_LIST context를 MOD_DOWNLOAD_TARGETS에 결합
# =========================================================
CONTEXT_MAP_KEYS = [
    '_context_root',
    '_context_wafer',
    '_context_process',
    '_context_step',
    '_context_ppid',
    '_context_mod_path',
]


context_source = matched_context_source[
    [
        SOURCE_ROOT_COL,
        SOURCE_WAFER_COL,
        SOURCE_PROCESS_COL,
        SOURCE_STEP_COL,
        SOURCE_CONTEXT_COLUMN_MAP['ppid_main_ovl'],
        SOURCE_MOD_PATH_COL,
        *[
            SOURCE_CONTEXT_COLUMN_MAP[column]
            for column in CONTEXT_COLUMNS
            if column != 'ppid_main_ovl'
        ],
    ]
].copy()

context_source = context_source.rename(columns={
    SOURCE_ROOT_COL: '_context_root',
    SOURCE_WAFER_COL: '_context_wafer',
    SOURCE_PROCESS_COL: '_context_process',
    SOURCE_STEP_COL: '_context_step',
    SOURCE_CONTEXT_COLUMN_MAP['ppid_main_ovl']: '_context_ppid',
    SOURCE_MOD_PATH_COL: '_context_mod_path',
    **{
        SOURCE_CONTEXT_COLUMN_MAP[column]: column
        for column in CONTEXT_COLUMNS
        if column != 'ppid_main_ovl'
    },
})

# mapping 결과에도 ppid_main_ovl을 원래 이름으로 남긴다.
context_source['ppid_main_ovl'] = context_source['_context_ppid']

context_source['_context_root'] = clean_id_series(
    context_source['_context_root'], upper=True
)
context_source['_context_wafer'] = clean_id_series(
    context_source['_context_wafer']
)

for column in [
    '_context_process',
    '_context_step',
    '_context_ppid',
    *CONTEXT_COLUMNS,
]:
    context_source[column] = clean_text_series(context_source[column])

context_source['_context_mod_path'] = clean_remote_path_series(
    context_source['_context_mod_path']
)

context_source = context_source.dropna(
    subset=CONTEXT_MAP_KEYS
).drop_duplicates()

context_mapping_records = []

for context_key, context_group in context_source.groupby(
    CONTEXT_MAP_KEYS,
    dropna=False,
    sort=False,
):
    if not isinstance(context_key, tuple):
        context_key = (context_key,)

    context_record = dict(zip(CONTEXT_MAP_KEYS, context_key))
    context_record.update(collapse_context_group(context_group).to_dict())
    context_mapping_records.append(context_record)

context_mapping = pd.DataFrame(context_mapping_records)


ovl_targets['_context_root'] = clean_id_series(
    ovl_targets[TARGET_ROOT_COL], upper=True
)
ovl_targets['_context_wafer'] = clean_id_series(
    ovl_targets[TARGET_WAFER_COL]
)
ovl_targets['_context_process'] = clean_text_series(
    ovl_targets[TARGET_PROCESS_COL]
)
ovl_targets['_context_step'] = clean_text_series(
    ovl_targets[TARGET_STEP_COL]
)
ovl_targets['_context_ppid'] = clean_text_series(
    ovl_targets[TARGET_PPID_COL]
)
ovl_targets['_context_mod_path'] = clean_remote_path_series(
    ovl_targets[TARGET_MOD_PATH_COL]
)

# 재실행 또는 수정된 2단계 target에 context가 이미 있어도 mapping 결과로 통일한다.
for column in [
    *CONTEXT_COLUMNS,
    'context_mapping_status',
    'context_ambiguous_columns',
]:
    if column in ovl_targets.columns:
        ovl_targets = ovl_targets.drop(columns=column)

ovl_targets = pd.merge(
    ovl_targets,
    context_mapping,
    on=CONTEXT_MAP_KEYS,
    how='left',
    validate='many_to_one',
)

ovl_targets['context_mapping_status'] = (
    ovl_targets['context_mapping_status'].fillna('NO_MAPPING')
)

ovl_targets['context_ambiguous_columns'] = (
    ovl_targets['context_ambiguous_columns'].fillna('')
)


context_status_summary = (
    ovl_targets['context_mapping_status']
    .value_counts(dropna=False)
    .rename_axis('context_mapping_status')
    .reset_index(name='target_rows')
)

print('\ncontext 매핑 결과')
display(context_status_summary)

missing_context_columns = [
    column for column in CONTEXT_COLUMNS if column not in ovl_targets.columns
]

if missing_context_columns:
    raise KeyError(
        f'다운로드 대상에 context column이 없습니다: {missing_context_columns}'
    )


# =========================================================
# 6. 대상 wafer key 정리
# =========================================================
ovl_targets[TARGET_ROOT_COL] = clean_id_series(
    ovl_targets[TARGET_ROOT_COL], upper=True
)
ovl_targets[TARGET_WAFER_COL] = clean_id_series(
    ovl_targets[TARGET_WAFER_COL]
)
ovl_targets[TARGET_MOD_PATH_COL] = clean_remote_path_series(
    ovl_targets[TARGET_MOD_PATH_COL]
)

target_dedup_columns = [
    TARGET_ROOT_COL,
    TARGET_WAFER_COL,
    TARGET_PROCESS_COL,
    TARGET_STEP_COL,
    TARGET_PPID_COL,
    TARGET_MOD_PATH_COL,
    *CONTEXT_COLUMNS,
    'context_mapping_status',
]
target_dedup_columns = list(dict.fromkeys(target_dedup_columns))

ovl_targets = (
    ovl_targets
    .dropna(
        subset=[
            TARGET_ROOT_COL,
            TARGET_WAFER_COL,
            TARGET_MOD_PATH_COL,
        ]
    )
    .drop_duplicates(subset=target_dedup_columns)
    .reset_index(drop=True)
)

if ovl_targets.empty:
    raise ValueError('정리 후 남아 있는 다운로드 대상이 없습니다.')

print('\n다운로드 대상')
print(
    'wafer:',
    ovl_targets[[TARGET_ROOT_COL, TARGET_WAFER_COL]]
    .drop_duplicates()
    .shape[0],
)
print('_mod_ 파일:', ovl_targets[TARGET_MOD_PATH_COL].nunique())


# =========================================================
# 7. FTP 다운로드 및 wafer/context 결합
# =========================================================
ftp = connect_ftp_stage3()

downloaded_frames = []
download_logs = []

try:
    grouped_targets = ovl_targets.groupby(
        TARGET_MOD_PATH_COL,
        sort=False,
    )
    total_files = ovl_targets[TARGET_MOD_PATH_COL].nunique()

    for file_no, (remote_path, target_group) in enumerate(
        grouped_targets,
        start=1,
    ):
        remote_path = str(remote_path)

        target_keys = (
            target_group[[TARGET_ROOT_COL, TARGET_WAFER_COL]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        requested_wafers = len(target_keys)
        cache_path = make_cache_path(
            remote_path,
            target_group,
            TARGET_ROOT_COL,
            TARGET_WAFER_COL,
        )

        print('\n' + '=' * 80)
        print(f'[{file_no}/{total_files}] {remote_path}')
        print(f'요청 wafer: {requested_wafers:,}')

        if cache_path.exists():
            filtered = pd.read_pickle(cache_path)

            cache_missing_context = [
                column
                for column in CONTEXT_COLUMNS
                if column not in filtered.columns
            ]

            if cache_missing_context:
                print(
                    'context 없는 cache 무시:',
                    cache_missing_context,
                )
            else:
                downloaded_frames.append(filtered)
                found_wafers = (
                    filtered[['root_lot_id_key', 'wafer_id_key']]
                    .drop_duplicates()
                    .shape[0]
                    if {
                        'root_lot_id_key',
                        'wafer_id_key',
                    }.issubset(filtered.columns)
                    else 0
                )

                download_logs.append({
                    'remote_path': remote_path,
                    'status': 'CACHE',
                    'requested_wafers': requested_wafers,
                    'found_wafers': found_wafers,
                    'raw_rows': pd.NA,
                    'filtered_rows': len(filtered),
                    'cache_path': str(cache_path),
                    'error': '',
                })
                print(f'context cache 사용: {len(filtered):,} rows')
                continue

        raw_ovl = None
        last_error = None

        for attempt in range(1, FTP_RETRY + 1):
            try:
                raw_ovl = download_csv_from_ftp(ftp, remote_path)
                last_error = None
                break

            except Exception as error:
                last_error = error
                print(
                    f'다운로드 실패 {attempt}/{FTP_RETRY}: {error}'
                )
                close_ftp_stage3(ftp)

                if attempt < FTP_RETRY:
                    time.sleep(2)
                    ftp = connect_ftp_stage3()

        if raw_ovl is None:
            download_logs.append({
                'remote_path': remote_path,
                'status': 'FAILED',
                'requested_wafers': requested_wafers,
                'found_wafers': 0,
                'raw_rows': 0,
                'filtered_rows': 0,
                'cache_path': str(cache_path),
                'error': str(last_error),
            })
            continue

        raw_rows = len(raw_ovl)

        raw_root_col = find_column_ci(
            raw_ovl,
            ['root_lot_id', 'root_lot_id_ovl'],
        )
        raw_wafer_col = find_column_ci(
            raw_ovl,
            ['wafer_id', 'wafer_id_ovl'],
        )

        raw_ovl['_root_key'] = clean_id_series(
            raw_ovl[raw_root_col], upper=True
        )
        raw_ovl['_wafer_key'] = clean_id_series(
            raw_ovl[raw_wafer_col]
        )

        # 2단계 MATCHED_LIST context를 기준으로 통일한다.
        for column in [
            *CONTEXT_COLUMNS,
            'context_mapping_status',
            'context_ambiguous_columns',
            'selected_process_id',
            'selected_step_seq',
            'selected_ppid_main_ovl',
        ]:
            if column in raw_ovl.columns:
                raw_ovl = raw_ovl.drop(columns=column)

        target_meta_source = target_group.copy()
        target_meta_source['_root_key'] = clean_id_series(
            target_meta_source[TARGET_ROOT_COL], upper=True
        )
        target_meta_source['_wafer_key'] = clean_id_series(
            target_meta_source[TARGET_WAFER_COL]
        )

        target_meta_records = []

        for target_key, wafer_target_group in target_meta_source.groupby(
            ['_root_key', '_wafer_key'],
            dropna=False,
            sort=False,
        ):
            if not isinstance(target_key, tuple):
                target_key = (target_key,)

            target_record = dict(
                zip(['_root_key', '_wafer_key'], target_key)
            )
            target_record.update(
                aggregate_target_meta(
                    wafer_target_group,
                    TARGET_PROCESS_COL,
                    TARGET_STEP_COL,
                    TARGET_PPID_COL,
                ).to_dict()
            )
            target_meta_records.append(target_record)

        target_meta = pd.DataFrame(target_meta_records)

        filtered = pd.merge(
            raw_ovl,
            target_meta,
            on=['_root_key', '_wafer_key'],
            how='inner',
            validate='many_to_one',
        )

        found_wafers = (
            filtered[['_root_key', '_wafer_key']]
            .drop_duplicates()
            .shape[0]
        )

        filtered['source_ovl_mod_path'] = remote_path
        filtered['source_download_time'] = datetime.now()
        filtered['context_cache_version'] = CACHE_VERSION
        filtered['root_lot_id_key'] = filtered['_root_key']
        filtered['wafer_id_key'] = filtered['_wafer_key']
        filtered['rlid_wid'] = (
            filtered['root_lot_id_key']
            + '_'
            + filtered['wafer_id_key']
        )

        filtered = filtered.drop(
            columns=['_root_key', '_wafer_key']
        )

        context_missing_after_merge = [
            column
            for column in CONTEXT_COLUMNS
            if column not in filtered.columns
        ]

        if context_missing_after_merge:
            raise KeyError(
                f'FTP 결합 후 context column 누락: '
                f'{context_missing_after_merge}'
            )

        filtered.to_pickle(cache_path)
        downloaded_frames.append(filtered)

        download_logs.append({
            'remote_path': remote_path,
            'status': 'DOWNLOADED',
            'requested_wafers': requested_wafers,
            'found_wafers': found_wafers,
            'raw_rows': raw_rows,
            'filtered_rows': len(filtered),
            'cache_path': str(cache_path),
            'error': '',
        })

        print(f'전체 row: {raw_rows:,}')
        print(f'필터 row: {len(filtered):,}')
        print(f'확인 wafer: {found_wafers:,}/{requested_wafers:,}')

finally:
    close_ftp_stage3(ftp)


# =========================================================
# 8. 결과 통합
# =========================================================
df_ovl_download_log = pd.DataFrame(download_logs)

if not downloaded_frames:
    failed_log_path = OVL_DOWNLOAD_DIR / (
        'OVL_CONTEXT_DOWNLOAD_FAILED_LOG.xlsx'
    )
    df_ovl_download_log.to_excel(failed_log_path, index=False)
    raise ValueError(
        '다운로드된 OVL 데이터가 없습니다.\n'
        f'로그: {failed_log_path}'
    )

ovl_filtered_all = pd.concat(
    downloaded_frames,
    ignore_index=True,
)

ovl_filtered_all = (
    ovl_filtered_all
    .drop_duplicates()
    .reset_index(drop=True)
)


missing_final_context = [
    column
    for column in CONTEXT_COLUMNS
    if column not in ovl_filtered_all.columns
]

if missing_final_context:
    raise KeyError(
        f'통합 OVL 데이터에 context column이 없습니다: {missing_final_context}'
    )


# =========================================================
# 9. 저장
# =========================================================
timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

OVL_PICKLE_PATH = OVL_DOWNLOAD_DIR / (
    f'OVL_FILTERED_ALL__CONTEXT__{timestamp}.pkl'
)
OVL_LOG_PATH = OVL_DOWNLOAD_DIR / (
    f'OVL_DOWNLOAD_LOG__CONTEXT__{timestamp}.xlsx'
)

ovl_filtered_all.to_pickle(OVL_PICKLE_PATH)

final_context_summary = (
    ovl_filtered_all
    .groupby(
        CONTEXT_COLUMNS + ['context_mapping_status'],
        dropna=False,
        as_index=False,
    )
    .agg(
        rows=('rlid_wid', 'size'),
        wafers=('rlid_wid', 'nunique'),
    )
)

with pd.ExcelWriter(OVL_LOG_PATH, engine='openpyxl') as writer:
    df_ovl_download_log.to_excel(
        writer,
        sheet_name='DOWNLOAD_LOG',
        index=False,
    )
    context_status_summary.to_excel(
        writer,
        sheet_name='TARGET_CONTEXT_STATUS',
        index=False,
    )
    context_mapping.to_excel(
        writer,
        sheet_name='CONTEXT_MAPPING',
        index=False,
    )
    final_context_summary.to_excel(
        writer,
        sheet_name='FINAL_CONTEXT_SUMMARY',
        index=False,
    )

if SAVE_COMBINED_CSV:
    OVL_CSV_PATH = OVL_DOWNLOAD_DIR / (
        f'OVL_FILTERED_ALL__CONTEXT__{timestamp}.csv'
    )
    ovl_filtered_all.to_csv(
        OVL_CSV_PATH,
        index=False,
        encoding='utf-8-sig',
    )


# =========================================================
# 10. 결과 확인
# =========================================================
target_wafer_count = (
    ovl_targets[[TARGET_ROOT_COL, TARGET_WAFER_COL]]
    .drop_duplicates()
    .shape[0]
)

downloaded_wafer_count = (
    ovl_filtered_all[['root_lot_id_key', 'wafer_id_key']]
    .drop_duplicates()
    .shape[0]
)

print('\n' + '=' * 90)
print('3단계 context 보존 다운로드 완료')
print(f'요청 wafer: {target_wafer_count:,}')
print(f'다운로드 완료 wafer: {downloaded_wafer_count:,}')
print(f'통합 OVL row: {len(ovl_filtered_all):,}')
print(f'OVL Pickle: {OVL_PICKLE_PATH}')
print(f'로그: {OVL_LOG_PATH}')
print(f'보존 context: {CONTEXT_COLUMNS}')
print('=' * 90)

display(df_ovl_download_log)
display(final_context_summary)
