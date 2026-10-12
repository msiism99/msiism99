# 1단계 수정본
# Wafer-level UFI와 FTP OVL list 매칭
#
# 주요 수정:
#   1. UFI item을 하나씩 조회한 후 concat
#   2. data.apply(pd.to_numeric, errors='ignore') 제거
#   3. category column을 비교 전에 string/datetime으로 명시 변환
#   4. UFI tkout_time과 FTP 수정시간을 datetime64로 강제 변환
#   5. 실패 시 정확한 오류 줄을 traceback으로 출력
# =========================================================
import warnings
warnings.filterwarnings('default')

import time
import datetime
import posixpath
import traceback

from ftplib import FTP
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import bigdataquery as bdq


# =========================================================
# 0. FTP 설정
# =========================================================
ftp_host = ***
ftp_port = ***
ftp_user = ***
ftp_password = ***

FTP_ROOT_DIR = '/PHOTO_OVERLAY_BB/database'
FTP_TIMEOUT = 120
FTP_PASSIVE_MODE = True


# =========================================================
# 1. UFI 및 OVL 조회 기간
# =========================================================
# UFI 측정일: 2026-08-03 00:00 이상, 2026-08-10 00:00 미만
UFI_DATE_FROM = pd.Timestamp('2026-09-28')
UFI_DATE_TO = pd.Timestamp('2026-10-04')
UFI_DATE_TO_EXCLUSIVE = UFI_DATE_TO + pd.Timedelta(days=1)

# OVL list 파일은 UFI보다 먼저 생성되므로 과거 150일까지 검색
OVL_LOOKBACK_DAYS = 150
OVL_FILE_DATE_FROM = UFI_DATE_FROM - pd.Timedelta(days=OVL_LOOKBACK_DAYS)
OVL_FILE_DATE_TO = pd.Timestamp.now()

TAG_WEEKNUM = '2640'


# =========================================================
# 2. 결과 저장 설정
# =========================================================
OUTPUT_DIR = Path.cwd() / 'UFI_OVL_OUTPUT'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOW_STR = datetime.datetime.now().strftime('%y%m%d_%H%M')
OUTPUT_PATH = OUTPUT_DIR / (
    f'UFI_OVL_MATCH_LIST__{TAG_WEEKNUM}__{NOW_STR}.xlsx'
)


# =========================================================
# 3. 선택 LOT 설정
# =========================================================
list_root_lot_id_dict = {
    'VMG_GPS2.0': sorted(set([
        'GON269', 'GOO124', 'GOO020', 'GON889', 'GON565',
        'GON378', 'GOO122', 'GOO020', 'GOO124'
    ]))
}


# =========================================================
# 4. UFI 조건
# =========================================================
# 필요한 조건만 활성화해서 사용하면 된다.
rows_ufi = [
    [['PFB3'], ['P3UN'], ['UN048217'], ['UFI1078']],
    [['PFB3'], ['P3UN'], ['UN077263'], ['UFI1027', 'UFI1053']],
    [['PFB3'], ['P3UN'], ['UN106435'], ['UFI1064', 'UFI1035']],

    [['KFBG'], ['KGVM'], ['VM077251_G2D'], ['UFI1019']],            # 1019 iBBD / 1020 DOC / 9019 iBBD + DOC / 1024 GBC NOP
    [['KFBG'], ['KGVH'], ['VH077251_G2D'], ['UFI1019']],
    [['KFBH'], ['KHVF'], ['VF077251_G2D'], ['UFI1019']],
    [['PFBB'], ['PBVS'], ['VS077251_G2D'], ['UFI1019']],
    [['PFBB'], ['PBVC'], ['VC077251_G2D'], ['UFI1019']],

    [['KFBG'], ['KGVM'], ['VM100433'], ['UFI4444', 'UFI1032']],     # 9999 LNC / 4444 LNC(BLK 제외) / 1032 직선 LBR / 1047 사선 LBR
    [['KFBG'], ['KGVH'], ['VH100433'], ['UFI4444', 'UFI1032']],
    [['KFBH'], ['KHVF'], ['VF100433'], ['UFI4444', 'UFI1032']],
    [['PFBB'], ['PBVS'], ['VS100433'], ['UFI4444', 'UFI1032']],
    [['PFBB'], ['PBVC'], ['VC100433'], ['UFI4444', 'UFI1032']],
]

PPID_MAIN_OVL_FILTERS = {
    ('PBVS', 'VS077251_G2D'): ['3G2PBVS.VS075P_AI','3G2PBVS.VS075P_B11'],
    ('PBVS', 'VS100433'): ['3ETPBVS.VS100P_DS', '3ETPBVS.VS100P_J9'],
    ('PBVC', 'VC077251_G2D'): ['3G2PBVC.VC075P_PV10','3G2PBVC.VC075P_B11'],
    ('PBVC', 'VC100433'): ['3E9PBVC.VC100P_DES','3E1PBVC.VC100P_J9'],

    ('KGVH', 'VH077251_G2D'): ['3GJKGVH.VH075P_R7527', '3G2KGVH.VH075P_R7527', '3G2KGVH.VH075P_C',],
    ('KGVH', 'VH100433'): ['3ETKGVH.VH100P_J9'],
    ('KGVM', 'VM077251_G2D'): ['3G2KGVM.VM075P_R7527'],
    ('KGVM', 'VM100433'): ['3E9KGVM.VM100P_GBL', '3ETKGVM.VM100P_J9'],
    ('KHVF', 'VF077251_G2D'): ['3G2KHVF.VF075P_TES'],
    ('KHVF', 'VF100433'): ['3ETKHVF.VF100P_DS'],

    ('P3UN', 'UN048217'): ['3EXP3UN.UN046P_R93S'],
    ('P3UN', 'UN077263'): ['3FAP3UN.UN075P_IP'],
    ('P3UN', 'UN106435'): ['33WP3UN.UN100P_HE1D', '33WP3UN.UN100P'],
    
    # 여러 PPID를 허용하는 경우
    # ('KGVM', 'VM077251_G2D'): [
    #     'PPID_A',
    #     'PPID_B',
    # ],
}

# =========================================================
# 5. 공통 함수
# =========================================================
def categorical_to_string(df):
    """category column은 비교·병합 전에 pandas string으로 변환한다."""
    result = df.copy()

    for column in result.columns:
        if isinstance(result[column].dtype, pd.CategoricalDtype):
            result[column] = result[column].astype('string')

    return result


def clean_id_series(series):
    return (
        series
        .astype('string')
        .str.strip()
        .str.replace(r'\.0$', '', regex=True)
    )


def normalize_key_columns(df):
    result = categorical_to_string(df)

    for column in ['root_lot_id', 'lot_id', 'wafer_id']:
        if column in result.columns:
            result[column] = clean_id_series(result[column])

    if {'root_lot_id', 'wafer_id'}.issubset(result.columns):
        result['rlid_wid'] = (
            result['root_lot_id'] + '_' + result['wafer_id']
        )

    return result


def normalize_datetime_series(series):
    """category/object/datetime 입력을 비교 가능한 datetime64로 통일한다."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype('string')

    return pd.to_datetime(series, errors='coerce')


def select_existing_columns(df, candidates):
    return [column for column in candidates if column in df.columns]


def add_suffix_except_keys(df, suffix, keys):
    rename_map = {
        column: f'{column}_{suffix}'
        for column in df.columns
        if column not in keys
    }
    return df.rename(columns=rename_map)


def add_config_columns(df, config):
    result = df.copy()

    for column, value in reversed(list(config.items())):
        result.insert(0, column, value)

    return result


def normalize_item_ids(item_ids):
    if isinstance(item_ids, str):
        item_ids = [item_ids]

    return sorted({
        str(item_id).strip().upper()
        for item_id in item_ids
        if str(item_id).strip()
    })


# =========================================================
# 6. FTP 함수
# =========================================================
def connect_ftp():
    ftp = FTP()
    ftp.connect(ftp_host, ftp_port, timeout=FTP_TIMEOUT)
    ftp.login(user=ftp_user, passwd=ftp_password)
    ftp.set_pasv(FTP_PASSIVE_MODE)
    ftp.encoding = 'utf-8'

    return ftp


def parse_ftp_modify_time(value):
    if not value:
        return pd.NaT

    try:
        parsed = datetime.datetime.strptime(
            str(value)[:14],
            '%Y%m%d%H%M%S'
        )
        return pd.Timestamp(parsed)
    except Exception:
        return pd.NaT


def normalize_remote_path(remote_dir, name):
    if name.startswith('/'):
        return name

    if name.startswith(remote_dir.rstrip('/') + '/'):
        return name

    return posixpath.join(remote_dir, name)


def list_ftp_csv_files(ftp, remote_dir):
    records = []

    # -----------------------------------------------------
    # MLSD 방식
    # -----------------------------------------------------
    try:
        for name, facts in ftp.mlsd(
            remote_dir,
            facts=['type', 'modify', 'size']
        ):
            if facts.get('type') != 'file':
                continue

            filename = str(name).lower()

            if 'list' not in filename or not filename.endswith('.csv'):
                continue

            records.append({
                'ovl_list_path': posixpath.join(remote_dir, name),
                'ovl_file_mtime': parse_ftp_modify_time(
                    facts.get('modify')
                ),
                'ovl_file_size': pd.to_numeric(
                    facts.get('size'),
                    errors='coerce'
                )
            })

        return records

    except Exception as error:
        print('[FTP] MLSD 사용 불가 → NLST 방식 사용:', error)

    # -----------------------------------------------------
    # NLST + MDTM fallback
    # -----------------------------------------------------
    for name in ftp.nlst(remote_dir):
        remote_path = normalize_remote_path(remote_dir, name)
        filename = posixpath.basename(remote_path).lower()

        if 'list' not in filename or not filename.endswith('.csv'):
            continue

        modified_time = pd.NaT

        try:
            response = ftp.sendcmd(f'MDTM {remote_path}')

            if response.startswith('213'):
                modified_time = parse_ftp_modify_time(
                    response.split()[-1]
                )
        except Exception:
            pass

        records.append({
            'ovl_list_path': remote_path,
            'ovl_file_mtime': modified_time,
            'ovl_file_size': np.nan
        })

    return records


def read_csv_from_ftp(ftp, remote_path):
    buffer = BytesIO()
    ftp.retrbinary(f'RETR {remote_path}', buffer.write)

    raw_data = buffer.getvalue()
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                BytesIO(raw_data),
                encoding=encoding,
                on_bad_lines='skip',
                low_memory=False
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f'CSV encoding 확인 실패: {remote_path} / {last_error}'
    )


# =========================================================
# 7. Wafer-level UFI 조회
# =========================================================
def query_single_wafer_ufi(
    tag_line_id,
    tag_process_id,
    item_id,
    selected_root_lots=None
):
    """다중 item category 문제를 피하기 위해 UFI item 하나씩 조회한다."""
    params = {
        'table_name': 'eds.m_eds_wf_measure',
        'dateFrom': UFI_DATE_FROM.strftime('%Y-%m-%d'),
        'dateTo': UFI_DATE_TO_EXCLUSIVE.strftime('%Y-%m-%d'),
        'step_seq': 'EDS_P',
        'line_id': tag_line_id,
        'process_id': tag_process_id,
        'item_id': [item_id],
        'subitem_id': 'A_CSUM_PERCENT',
        'setLimit': False
    }

    if selected_root_lots:
        params['root_lot_id'] = list(selected_root_lots)

    print(
        f'[UFI QUERY ITEM] {tag_line_id} / {tag_process_id} / '
        f'{item_id} / {UFI_DATE_FROM.date()} ~ {UFI_DATE_TO.date()}'
    )

    data = bdq.getData(param=params)

    if data is None or len(data) == 0:
        print(f'[UFI QUERY ITEM] {item_id}: 0 rows')
        return pd.DataFrame()

    result = categorical_to_string(data)
    result['_requested_item_id'] = item_id

    print(f'[UFI QUERY ITEM] {item_id}: {len(result):,} rows')

    return result


def load_wafer_ufi(
    tag_line_id,
    tag_process_id,
    tag_ufi_id,
    selected_root_lots=None
):
    item_ids = normalize_item_ids(tag_ufi_id)
    query_frames = []

    for item_id in item_ids:
        item_data = query_single_wafer_ufi(
            tag_line_id=tag_line_id,
            tag_process_id=tag_process_id,
            item_id=item_id,
            selected_root_lots=selected_root_lots
        )

        if not item_data.empty:
            query_frames.append(item_data)

    if not query_frames:
        return pd.DataFrame()

    # 전체 column을 pd.to_numeric으로 변환하지 않는다.
    df = pd.concat(query_frames, ignore_index=True, sort=False)
    df = normalize_key_columns(df)

    required = {'root_lot_id', 'wafer_id', 'tkout_time'}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f'Wafer UFI 필수 컬럼 없음: {sorted(missing)}')

    # 핵심 수정: category가 남아 있어도 datetime64로 강제 변환
    df['tkout_time'] = normalize_datetime_series(df['tkout_time'])

    invalid_time_rows = int(df['tkout_time'].isna().sum())

    if invalid_time_rows:
        print(
            f'[UFI QUERY] tkout_time 변환 실패 row: '
            f'{invalid_time_rows:,}'
        )

    # 정확히 UFI_DATE_FROM 이상, UFI_DATE_TO_EXCLUSIVE 미만
    time_mask = (
        df['tkout_time'].ge(UFI_DATE_FROM)
        & df['tkout_time'].lt(UFI_DATE_TO_EXCLUSIVE)
    )
    df = df.loc[time_mask].copy()

    if selected_root_lots:
        selected_root_lots_clean = set(
            clean_id_series(pd.Series(selected_root_lots))
            .dropna()
            .tolist()
        )
        df = df[
            df['root_lot_id'].isin(selected_root_lots_clean)
        ].copy()

    # 동일 wafer/item이 여러 번 존재하면 가장 최근 결과 사용
    dedup_keys = select_existing_columns(
        df,
        [
            'root_lot_id',
            'lot_id',
            'wafer_id',
            'item_id',
            '_requested_item_id',
            'subitem_id'
        ]
    )

    if dedup_keys:
        df = (
            df
            .sort_values('tkout_time')
            .drop_duplicates(dedup_keys, keep='last')
        )

    print(
        f"[UFI QUERY] rows={len(df):,}, "
        f"wafers={df['rlid_wid'].nunique():,}, "
        f"root lots={df['root_lot_id'].nunique():,}"
    )

    return df.reset_index(drop=True)


# =========================================================
# 8. FTP OVL list 조회
# =========================================================
def load_ovl_list(tag_process_id, tag_step_seq, target_root_lots):
    target_root_lots = set(
        clean_id_series(pd.Series(target_root_lots))
        .dropna()
        .tolist()
    )
    remote_dir = posixpath.join(
        FTP_ROOT_DIR,
        tag_process_id,
        tag_step_seq
    )

    print('[FTP] 연결 중:', ftp_host)
    ftp = connect_ftp()

    try:
        print('[FTP] OVL 폴더 검색:', remote_dir)
        file_records = list_ftp_csv_files(ftp, remote_dir)

        if not file_records:
            print('[FTP] list CSV 없음:', remote_dir)
            return pd.DataFrame()

        file_info = (
            pd.DataFrame(file_records)
            .drop_duplicates('ovl_list_path')
        )

        # 핵심 수정: between() 전에 category/object를 datetime64로 강제
        file_info['ovl_file_mtime'] = normalize_datetime_series(
            file_info['ovl_file_mtime']
        )

        has_mtime = file_info['ovl_file_mtime'].notna()
        valid_date = file_info['ovl_file_mtime'].between(
            OVL_FILE_DATE_FROM,
            OVL_FILE_DATE_TO,
            inclusive='both'
        )
        file_info = file_info[
            ~has_mtime | valid_date
        ].reset_index(drop=True)

        print(
            f'[FTP] 검색 대상 list 파일={len(file_info):,}, '
            f'기간={OVL_FILE_DATE_FROM.date()}~'
            f'{OVL_FILE_DATE_TO.date()}'
        )

        frames = []

        for file_no, file_row in enumerate(
            file_info.itertuples(index=False),
            start=1
        ):
            remote_path = file_row.ovl_list_path

            try:
                print(
                    f'[FTP] {file_no}/{len(file_info)} 읽기: '
                    f'{remote_path}'
                )

                part = read_csv_from_ftp(ftp, remote_path)
                part = normalize_key_columns(part)

                if not {'root_lot_id', 'wafer_id'}.issubset(part.columns):
                    print('[FTP] key 컬럼 없음:', remote_path)
                    continue

                # 해당 주 UFI에 존재하는 root lot만 유지
                part = part[
                    part['root_lot_id'].isin(target_root_lots)
                ].copy()

                if part.empty:
                    continue

                part['ovl_list_path'] = remote_path
                part['ovl_mod_path'] = remote_path.replace(
                    '_list_',
                    '_mod_'
                )
                part['ovl_file_mtime'] = file_row.ovl_file_mtime
                part['ovl_file_size'] = file_row.ovl_file_size

                for time_col in [
                    'tkout_time',
                    'tkout_time_main',
                    'tkout_time_base'
                ]:
                    if time_col in part.columns:
                        part[time_col] = normalize_datetime_series(
                            part[time_col]
                        )

                frames.append(part)

            except Exception:
                print('[FTP] CSV 읽기 실패:', remote_path)
                traceback.print_exc()

        if not frames:
            print('[FTP] UFI와 관련된 OVL list data 없음')
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True, sort=False)
        df['ovl_file_mtime'] = normalize_datetime_series(
            df['ovl_file_mtime']
        )

        # FTP에 동일한 OVL list가 중복되어 있으면 하나만 유지
        dedup_candidates = [
            'root_lot_id', 'lot_id', 'wafer_id',
            'tkout_time', 'tkout_time_main',
            'eqp_id', 'ppid',
            'eqp_id_main', 'ppid_main', 'reticle_id_main',
            'eqp_id_base', 'ppid_base', 'reticle_id_base'
        ]
        dedup_keys = select_existing_columns(df, dedup_candidates)

        if dedup_keys:
            df = (
                df
                .sort_values(
                    'ovl_file_mtime',
                    ascending=False,
                    na_position='last'
                )
                .drop_duplicates(dedup_keys, keep='first')
            )

        print(
            f"[FTP OVL LIST] rows={len(df):,}, "
            f"wafers={df['rlid_wid'].nunique():,}"
        )

        return df.reset_index(drop=True)

    finally:
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass


# =========================================================
# 9. UFI와 OVL 비교
# =========================================================
def compare_ufi_ovl(ufi_df, ovl_df, config):
    merge_keys = ['root_lot_id', 'wafer_id']

    ufi_candidates = [
        'root_lot_id', 'lot_id', 'wafer_id', 'rlid_wid',
        'line_id', 'process_id', 'step_seq',
        'test_type', 'lot_type', 'tkout_time',
        'item_type', 'item_id', '_requested_item_id',
        'subitem_id', 'value',
        'last_update_time', 'src_param_key'
    ]

    ovl_candidates = [
        'root_lot_id', 'lot_id', 'wafer_id', 'rlid_wid',
        'process_id', 'step_seq',
        'tkout_time', 'eqp_id', 'ppid',
        'tkout_time_main', 'eqp_id_main', 'ppid_main',
        'reticle_id_main',
        'tkout_time_base', 'eqp_id_base', 'ppid_base',
        'reticle_id_base',
        'ovl_list_path', 'ovl_mod_path',
        'ovl_file_mtime', 'ovl_file_size'
    ]

    ufi_small = ufi_df[
        select_existing_columns(ufi_df, ufi_candidates)
    ].copy()

    if ovl_df.empty:
        ovl_small = pd.DataFrame(columns=merge_keys)
    else:
        ovl_small = ovl_df[
            select_existing_columns(ovl_df, ovl_candidates)
        ].copy()

    ufi_small = add_suffix_except_keys(
        ufi_small,
        'ufi',
        merge_keys
    )
    ovl_small = add_suffix_except_keys(
        ovl_small,
        'ovl',
        merge_keys
    )

    comparison = pd.merge(
        ufi_small,
        ovl_small,
        on=merge_keys,
        how='left',
        indicator=True
    )

    comparison['match_status'] = np.where(
        comparison['_merge'].eq('both'),
        'MATCHED',
        'UFI_ONLY'
    )
    comparison = comparison.drop(columns='_merge')

    comparison['rlid_wid'] = (
        comparison['root_lot_id'].astype('string')
        + '_'
        + comparison['wafer_id'].astype('string')
    )
    comparison = add_config_columns(comparison, config)

    matched = comparison[
        comparison['match_status'].eq('MATCHED')
    ].copy()
    ufi_only = comparison[
        comparison['match_status'].eq('UFI_ONLY')
    ].copy()

    if ovl_df.empty:
        ovl_candidates_sheet = pd.DataFrame()
    else:
        ovl_candidates_sheet = add_config_columns(
            ovl_df.copy(),
            config
        )

    summary = {
        **config,
        'ufi_rows': len(ufi_df),
        'ufi_unique_wafers': ufi_df['rlid_wid'].nunique(),
        'ufi_unique_root_lots': ufi_df['root_lot_id'].nunique(),
        'ovl_candidate_rows': len(ovl_df),
        'ovl_unique_wafers': (
            ovl_df['rlid_wid'].nunique() if not ovl_df.empty else 0
        ),
        'matched_rows': len(matched),
        'matched_unique_wafers': matched['rlid_wid'].nunique(),
        'ufi_only_rows': len(ufi_only),
        'ufi_only_unique_wafers': ufi_only['rlid_wid'].nunique()
    }

    return (
        matched,
        ufi_only,
        ovl_candidates_sheet,
        pd.DataFrame([summary])
    )


# =========================================================
# 10. Excel 서식
# =========================================================
def format_excel(writer):
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = 'A2'
        worksheet.auto_filter.ref = worksheet.dimensions

        max_check_row = min(worksheet.max_row, 300)

        for column_index in range(1, worksheet.max_column + 1):
            cells = [
                worksheet.cell(
                    row=row,
                    column=column_index
                ).value
                for row in range(1, max_check_row + 1)
            ]

            max_length = max(
                [
                    len(str(value))
                    for value in cells
                    if value is not None
                ]
                or [10]
            )

            column_letter = worksheet.cell(
                row=1,
                column=column_index
            ).column_letter
            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 10),
                45
            )


# %%
# =========================================================
# 11. 실행
# =========================================================
matched_frames = []
ufi_only_frames = []
ovl_candidate_frames = []
summary_frames = []
error_records = []

for row in rows_ufi:
    print('\n' + '=' * 80)
    print('START:', row)
    print('=' * 80)

    try:
        tag_line_id = row[0][0]
        tag_process_id = row[1][0]
        tag_step_seq = row[2][0]
        tag_ufi_id = normalize_item_ids(row[3])

        tag_lot_selected = row[4] if len(row) >= 5 else ''
        selected_root_lots = (
            list_root_lot_id_dict[tag_lot_selected]
            if tag_lot_selected
            else []
        )

        config_id = (
            f"{tag_process_id}__{tag_step_seq}__"
            f"{'_'.join(tag_ufi_id)}__{tag_lot_selected}"
        )

        config = {
            'config_id': config_id,
            'target_line_id': tag_line_id,
            'target_process_id': tag_process_id,
            'target_step_seq': tag_step_seq,
            'target_ufi_ids': '_'.join(tag_ufi_id),
            'selected_lot_group': tag_lot_selected,
            'ufi_date_from': UFI_DATE_FROM.strftime('%Y-%m-%d'),
            'ufi_date_to': UFI_DATE_TO.strftime('%Y-%m-%d')
        }

        # -------------------------------------------------
        # 1. 지정 주차의 wafer-level UFI 조회
        # -------------------------------------------------
        ufi_df = load_wafer_ufi(
            tag_line_id=tag_line_id,
            tag_process_id=tag_process_id,
            tag_ufi_id=tag_ufi_id,
            selected_root_lots=selected_root_lots
        )

        if ufi_df.empty:
            print('⚠️ 해당 조건의 wafer-level UFI가 없습니다.')
            continue

        # -------------------------------------------------
        # 2. FTP OVL list CSV 검색
        # -------------------------------------------------
        ufi_root_lots = (
            ufi_df['root_lot_id']
            .dropna()
            .unique()
            .tolist()
        )

        ovl_df = load_ovl_list(
            tag_process_id=tag_process_id,
            tag_step_seq=tag_step_seq,
            target_root_lots=ufi_root_lots
        )

        # -------------------------------------------------
        # 3. root_lot_id + wafer_id 기준 일치
        # -------------------------------------------------
        matched, ufi_only, ovl_candidates, summary = compare_ufi_ovl(
            ufi_df=ufi_df,
            ovl_df=ovl_df,
            config=config
        )

        matched_frames.append(matched)
        ufi_only_frames.append(ufi_only)
        summary_frames.append(summary)

        if not ovl_candidates.empty:
            ovl_candidate_frames.append(ovl_candidates)

        print(
            f"[RESULT] matched wafers="
            f"{matched['rlid_wid'].nunique():,}, "
            f"UFI only wafers={ufi_only['rlid_wid'].nunique():,}"
        )

        time.sleep(2)

    except Exception as error:
        print('❌ 처리 실패:', row)
        traceback.print_exc()

        error_records.append({
            'row': repr(row),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
        })


# =========================================================
# 12. Excel 저장
# =========================================================
matched_all = (
    pd.concat(matched_frames, ignore_index=True, sort=False)
    if matched_frames
    else pd.DataFrame()
)

ufi_only_all = (
    pd.concat(ufi_only_frames, ignore_index=True, sort=False)
    if ufi_only_frames
    else pd.DataFrame()
)

ovl_candidates_all = (
    pd.concat(ovl_candidate_frames, ignore_index=True, sort=False)
    if ovl_candidate_frames
    else pd.DataFrame()
)

summary_all = (
    pd.concat(summary_frames, ignore_index=True, sort=False)
    if summary_frames
    else pd.DataFrame()
)

error_log = pd.DataFrame(error_records)

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    matched_all.to_excel(writer, sheet_name='MATCHED_LIST', index=False)
    ufi_only_all.to_excel( writer, sheet_name='UFI_ONLY', index=False)
    ovl_candidates_all.to_excel(writer, sheet_name='OVL_CANDIDATES', index=False)
    summary_all.to_excel(writer, sheet_name='SUMMARY', index=False)
    error_log.to_excel(writer, sheet_name='ERROR_LOG', index=False)
    format_excel(writer)

print('\n' + '=' * 80)
print('Excel 저장 완료')
print('파일:', OUTPUT_PATH)
print('MATCHED rows:', len(matched_all))
print(
    'MATCHED unique wafers:',
    matched_all['rlid_wid'].nunique()
    if not matched_all.empty
    else 0
)
print('ERROR rows:', len(error_log))
print('=' * 80)

