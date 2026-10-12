# 2단계 수정본
# =========================================================
# 2단계 V2 (DRM/xlwings 대응)
#
# - 정상 xlsx: pandas + openpyxl
# - 회사 DRM/암호화 xlsx: Excel COM + xlwings fallback
# - UFI_OVL_MATCH_LIST 파일만 입력 후보로 사용
# - 기존 PPID_MAIN_OVL_FILTERS가 메모리에 있으면 그대로 재사용
# - matched_ppid / mod_download_targets_ppid 변수 생성
# =========================================================
from pathlib import Path
from datetime import datetime

import pandas as pd


STAGE2_CODE_VERSION = 'STAGE2_V2_1_SKIP_EMPTY_GROUP_20260911'
print(f'\n[CODE VERSION] {STAGE2_CODE_VERSION}')


# =========================================================
# 1. 사용자 설정
# =========================================================
if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = Path.cwd() / 'UFI_OVL_OUTPUT'

OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 같은 kernel에서 기존 2단계를 실행했다면 당시 설정을 그대로 사용한다.
existing_ppid_filters = globals().get('PPID_MAIN_OVL_FILTERS')

if (
    isinstance(existing_ppid_filters, dict)
    and existing_ppid_filters
):
    PPID_MAIN_OVL_FILTERS = existing_ppid_filters.copy()
    print('메모리에 있는 기존 PPID_MAIN_OVL_FILTERS를 재사용합니다.')

else:
    # kernel을 재시작한 경우 아래 값을 실제 사용값으로 수정해야 한다.
    PPID_MAIN_OVL_FILTERS = {
        ('PBVS', 'VS077251_G2D'): [
            '3G2PBVS.VS075P_B76D5',
        ],
        ('PBVC', 'VC077251_G2D'): [
            '3G2PBVC.VC075P_PV10',
        ],
    }

    print(
        '기존 PPID filter 변수가 없어 코드의 기본값을 사용합니다.\n'
        'P3UN/KGVM/KGVH 등 추가 설정이 있었다면 '
        'PPID_MAIN_OVL_FILTERS에 다시 입력해야 합니다.'
    )


# =========================================================
# 2. 공통 함수
# =========================================================
def normalize_text(series):
    return series.astype('string').str.strip().str.upper()


def normalize_value(value):
    return str(value).strip().upper()


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        '다음 column 중 사용할 수 있는 column이 없습니다.\n'
        f'{candidates}\n\n'
        f'현재 column:\n{df.columns.tolist()}'
    )


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

        headers.append(
            base_name
            if duplicate_no == 0
            else f'{base_name}.{duplicate_no}'
        )

    return headers


def read_excel_sheet_robust(excel_path, sheet_name):
    """
    openpyxl을 먼저 사용하고, BadZipFile 등으로 실패하면
    설치된 Excel을 xlwings로 열어 sheet 값을 가져온다.
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
            f'openpyxl 실패: {type(openpyxl_error).__name__}: '
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
            f'openpyxl 실패: {type(openpyxl_error).__name__}: '
            f'{openpyxl_error}\n'
            f'xlwings 실패: {type(xlwings_error).__name__}: '
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
# 3. 정상적인 1단계 Excel 선택
# =========================================================
stage1_files = sorted(
    [
        path
        for path in OUTPUT_DIR.glob('UFI_OVL_MATCH_LIST__*.xlsx')
        if path.is_file()
        and not path.name.startswith('~$')
    ],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not stage1_files:
    raise FileNotFoundError(
        '1단계 UFI_OVL_MATCH_LIST Excel을 찾지 못했습니다.\n'
        f'검색 위치: {OUTPUT_DIR}'
    )

FIRST_EXCEL_PATH = None
matched_unfiltered = None
stage1_reader = None
stage1_file_errors = []

for candidate_path in stage1_files:
    try:
        if candidate_path.stat().st_size == 0:
            raise ValueError('0 byte file')

        candidate_data, candidate_reader = read_excel_sheet_robust(
            candidate_path,
            sheet_name='MATCHED_LIST',
        )

        if candidate_data.empty:
            raise ValueError('MATCHED_LIST sheet가 비어 있습니다.')

        FIRST_EXCEL_PATH = candidate_path
        matched_unfiltered = candidate_data
        stage1_reader = candidate_reader
        break

    except Exception as error:
        stage1_file_errors.append({
            'path': str(candidate_path),
            'size_bytes': candidate_path.stat().st_size,
            'error': f'{type(error).__name__}: {error}',
        })

        print(
            f'읽을 수 없는 1단계 Excel 건너뜀: '
            f'{candidate_path.name}\n'
            f'  {type(error).__name__}: {error}'
        )

if FIRST_EXCEL_PATH is None:
    error_details = '\n'.join(
        f'- {record["path"]} / '
        f'{record["size_bytes"]} bytes / '
        f'{record["error"]}'
        for record in stage1_file_errors
    )

    raise ValueError(
        '읽을 수 있는 1단계 Excel이 없습니다.\n\n'
        f'{error_details}'
    )

print(f'입력 Excel: {FIRST_EXCEL_PATH}')
print(f'Excel reader: {stage1_reader}')
print(f'원본 row: {len(matched_unfiltered):,}')


# =========================================================
# 4. 사용할 column 확인
# =========================================================
PROCESS_COL = find_column(
    matched_unfiltered,
    [
        'source_process_id_ovl',
        'process_id_ovl',
        'tag_process_id_ovl',
        'process_id',
    ]
)

STEP_COL = find_column(
    matched_unfiltered,
    [
        'source_step_seq_ovl',
        'step_seq_ovl',
        'tag_step_seq_ovl',
        'step_seq',
    ]
)

PPID_COL = 'ppid_main_ovl'

if PPID_COL not in matched_unfiltered.columns:
    raise KeyError(
        f'{PPID_COL} column이 없습니다.\n'
        f'PPID 관련 column: '
        f'{[column for column in matched_unfiltered.columns if "ppid" in column.lower()]}'
    )

MOD_PATH_COL = find_column(
    matched_unfiltered,
    ['ovl_mod_path_ovl', 'ovl_mod_path']
)

print('\n사용 column')
print(f'process : {PROCESS_COL}')
print(f'step    : {STEP_COL}')
print(f'PPID    : {PPID_COL}')
print(f'mod path: {MOD_PATH_COL}')


# =========================================================
# 5. 실제 제품/step/PPID 조합 확인
# =========================================================
ppid_check = matched_unfiltered[
    [PROCESS_COL, STEP_COL, PPID_COL]
].copy()

for column in [PROCESS_COL, STEP_COL, PPID_COL]:
    ppid_check[column] = normalize_text(ppid_check[column])

ppid_check = (
    ppid_check
    .value_counts(
        [PROCESS_COL, STEP_COL, PPID_COL],
        dropna=False
    )
    .rename('rows')
    .reset_index()
    .sort_values([PROCESS_COL, STEP_COL, PPID_COL])
    .reset_index(drop=True)
)

print('\n현재 Excel에 존재하는 PPID')
display(ppid_check)


# =========================================================
# 6. PPID 필터 설정 확인 및 적용
# =========================================================
if not PPID_MAIN_OVL_FILTERS:
    raise ValueError('PPID_MAIN_OVL_FILTERS가 비어 있습니다.')

empty_filters = [
    key
    for key, values in PPID_MAIN_OVL_FILTERS.items()
    if not values
]

if empty_filters:
    raise ValueError(
        f'PPID 목록이 비어 있는 그룹이 있습니다: {empty_filters}'
    )

matched_work = matched_unfiltered.copy()
matched_work['_process_key'] = normalize_text(matched_work[PROCESS_COL])
matched_work['_step_key'] = normalize_text(matched_work[STEP_COL])
matched_work['_ppid_key'] = normalize_text(matched_work[PPID_COL])

final_mask = pd.Series(False, index=matched_work.index)
filter_summary = []
filter_errors = []

for (process_id, step_seq), allowed_ppids in PPID_MAIN_OVL_FILTERS.items():
    process_key = normalize_value(process_id)
    step_key = normalize_value(step_seq)

    allowed_ppids = sorted({
        normalize_value(ppid)
        for ppid in allowed_ppids
        if str(ppid).strip()
    })

    group_mask = (
        matched_work['_process_key'].eq(process_key)
        & matched_work['_step_key'].eq(step_key)
    )

    selected_mask = (
        group_mask
        & matched_work['_ppid_key'].isin(allowed_ppids)
    )

    available_ppids = sorted(
        matched_work.loc[group_mask, '_ppid_key']
        .dropna()
        .unique()
        .tolist()
    )

    group_rows = int(group_mask.sum())
    selected_rows = int(selected_mask.sum())

    print('\n' + '-' * 70)
    print(f'GROUP: {process_key} / {step_key}')
    print(f'그룹 전체 row: {group_rows:,}')
    print(f'실제 PPID: {available_ppids}')
    print(f'설정 PPID: {allowed_ppids}')
    print(f'필터 통과 row: {selected_rows:,}')

    if group_rows == 0:
        # 이번 기간에 wafer가 없는 설정 그룹은 정상적으로 제외한다.
        filter_status = 'SKIPPED_NO_WAFER'
        print('처리 상태: 대상 wafer 없음 → 이번 실행에서 제외')
    elif selected_rows == 0:
        filter_status = 'ERROR_PPID_NOT_MATCHED'
        filter_errors.append(
            f'{process_key}/{step_key}: 설정 PPID {allowed_ppids}가 '
            f'실제 PPID {available_ppids}와 일치하지 않습니다.'
        )
    else:
        filter_status = 'SELECTED'

    filter_summary.append({
        'process_id': process_key,
        'step_seq': step_key,
        'configured_ppid': ', '.join(allowed_ppids),
        'available_ppid': ', '.join(available_ppids),
        'group_rows': group_rows,
        'selected_rows': selected_rows,
        'filter_status': filter_status,
    })

    final_mask |= selected_mask

df_filter_summary = pd.DataFrame(filter_summary)

if filter_errors:
    raise ValueError(
        '\nPPID 필터 설정을 확인해주세요.\n'
        + '\n'.join(f'- {error}' for error in filter_errors)
    )

matched_ppid = (
    matched_work.loc[final_mask]
    .drop(
        columns=['_process_key', '_step_key', '_ppid_key']
    )
    .reset_index(drop=True)
)

if matched_ppid.empty:
    raise ValueError('PPID 필터 결과가 0 row입니다.')


# =========================================================
# 7. _mod_ 다운로드 대상 생성
# =========================================================
target_columns = [PROCESS_COL, STEP_COL, PPID_COL]

for column in ['root_lot_id', 'wafer_id', 'rlid_wid']:
    if column in matched_ppid.columns:
        target_columns.append(column)

target_columns.append(MOD_PATH_COL)
target_columns = list(dict.fromkeys(target_columns))

mod_download_targets_ppid = (
    matched_ppid[target_columns]
    .dropna(subset=[MOD_PATH_COL])
    .drop_duplicates()
    .sort_values([PROCESS_COL, STEP_COL, PPID_COL, MOD_PATH_COL])
    .reset_index(drop=True)
)

mod_path_list_ppid = (
    matched_ppid[MOD_PATH_COL]
    .dropna()
    .drop_duplicates()
    .tolist()
)


# =========================================================
# 8. 신규 Excel 저장
# =========================================================
timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
PPID_OUTPUT_PATH = OUTPUT_DIR / f'PPID_FILTERED__{timestamp}.xlsx'

with pd.ExcelWriter(PPID_OUTPUT_PATH, engine='openpyxl') as writer:
    matched_ppid.to_excel(
        writer,
        sheet_name='MATCHED_LIST',
        index=False
    )
    mod_download_targets_ppid.to_excel(
        writer,
        sheet_name='MOD_DOWNLOAD_TARGETS',
        index=False
    )
    ppid_check.to_excel(
        writer,
        sheet_name='AVAILABLE_PPID',
        index=False
    )
    df_filter_summary.to_excel(
        writer,
        sheet_name='FILTER_SUMMARY',
        index=False
    )

print('\n' + '=' * 70)
print(f'원본 row: {len(matched_unfiltered):,}')
print(f'PPID 필터 결과 row: {len(matched_ppid):,}')
print(f'다운로드 대상 _mod_ 파일: {len(mod_path_list_ppid):,}개')
print(f'입력 Excel: {FIRST_EXCEL_PATH}')
print(f'필터 결과 저장: {PPID_OUTPUT_PATH}')
