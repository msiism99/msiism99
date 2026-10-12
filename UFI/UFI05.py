# 5단계 / 6단계
#
# CODE VERSION: STAGE5_STAGE6_CONTEXT_PRESERVED_V2_20260901
#
# 전제:
#   1~4단계를 실행하여 아래 DataFrame이 메모리에 존재하거나,
#   각 단계가 저장한 최신 Pickle 파일이 OUTPUT_DIR 아래에 존재해야 한다.
#
#   - ovl_filtered_all       : 측정 point별 OVL 원자료
#   - ufi_chip_filtered_all  : chip-level UFI 원자료
#
# 5단계:
#   안정화 4D RBF(fcp_x, fcp_y, ftp_x, ftp_y)
#   -> chip 내부 3x3 예측
#   -> chip 내부 실측 OVL 추가
#   -> 방향별/부호별 chip 대표값
#   -> chip-level UFI 결합
#
# 6단계:
#   OVL 0.1 nm bin별 UFI rate 계산 및 사발커브 저장


# %%
# =========================================================
# 공통 import
# =========================================================
from pathlib import Path
from datetime import datetime

import math
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

import bigdataquery as bdq


CODE_VERSION = 'STAGE5_STAGE6_CONTEXT_PRESERVED_V0901'
print(f'[CODE VERSION] {CODE_VERSION}')


# %%
# =========================================================
# 5단계 사용자 설정
# =========================================================
if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = Path.cwd() / 'UFI_OVL_OUTPUT'
else:
    OUTPUT_DIR = Path(OUTPUT_DIR)

RUN_TAG = str(TAG_WEEKNUM)

STAGE5_OUTPUT_DIR = OUTPUT_DIR / 'OVL_UFI_CHIP' / RUN_TAG
STAGE5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# RBF 설정
# ---------------------------------------------------------
# K의 대각선이 1이므로 lambda는 1에 대한 상대적인 크기다.
# 첫 비교 권장값: 1e-4, 1e-3, 1e-2
RBF_LAMBDA = 1e-3

# None이면 표준화된 4D 공간에서 k번째 최근접 거리의 중앙값으로 산출한다.
RBF_SIGMA = None
RBF_SIGMA_NEIGHBOR_K = 4
RBF_SIGMA_SCALE = 1.5

# 예측점의 가장 가까운 control point 거리가 sigma의 몇 배까지인지.
# 이를 초과하면 4D 외삽으로 판단하여 chip 대표값 계산에서 제외한다.
RBF_SUPPORT_SIGMA_MULTIPLIER = 2.5

# 원본 측정값 대비 비정상적으로 큰 prediction을 자르지 않고 INVALID 처리한다.
# 예: 측정 최대값이 2 nm라면 max(10, 5*2)=10 nm가 component별 한계.
PRED_LIMIT_FACTOR = 5.0
PRED_LIMIT_FLOOR_NM = 10.0

# condition number 계산은 큰 행렬에서 오래 걸리므로 이 point 수 이하만 계산.
RBF_CONDITION_CHECK_MAX_POINTS = 800

# predict 시 메모리 사용량 제한
RBF_PREDICT_CHUNK_SIZE = 20000


# ---------------------------------------------------------
# Chip 내부 3x3 sampling 설정
# ---------------------------------------------------------
# chip pitch 또는 chip 크기의 40%만큼 중심에서 이동:
# -0.4, 0, +0.4 -> 대략 chip의 10%, 50%, 90% 위치
CHIP_SAMPLE_EDGE_FRACTION = 0.40

# fab.m_chip_prod_wf_layout의 chip_center_x/y가 um이고,
# OVL fcp/ftp가 mm라는 기존 코드 기준. 단위가 같다면 1.0으로 변경한다.
LAYOUT_POSITION_SCALE_TO_OVL = 1.0 / 1000.0

# _mod_ 데이터에 STEP_PITCH_X/Y가 있으면 일반적으로 um 단위이므로
# fcp/ftp의 mm 단위로 변환한다. 해당 column이 이미 mm라면 1.0으로 변경.
SHOT_PITCH_SCALE_TO_OVL = 1.0 / 1000.0

# ftp = global point - shot center가 기본 좌표계.
# 장비/데이터 좌표계의 방향이 반대면 해당 축을 -1.0으로 변경한다.
FTP_X_SIGN = 1.0
FTP_Y_SIGN = 1.0

# 실측 point를 가장 가까운 chip에 배정할 때 chip pitch의 절반에 대한 여유.
ACTUAL_POINT_ASSIGN_TOLERANCE = 1.05

# Shot center 반경 기준. fcp_x/y가 mm 단위라는 전제다.
# sqrt(fcp_x**2 + fcp_y**2) > 140 mm이면 EDGE shot으로 구분한다.
EDGE_SHOT_RADIUS_MM = 140.0

# sample point 전체를 별도 Pickle로 저장한다.
SAVE_STAGE5_SAMPLE_POINTS = True


# ---------------------------------------------------------
# 2단계부터 마지막 chip-level 결과까지 보존할 PHOTO context
# ---------------------------------------------------------
# 6단계 사발커브의 그룹 기준에는 추가하지 않는다.
# 즉 기존 사발커브는 그대로 생성되고, 아래 column은 5단계 Pickle에만 남는다.
CONTEXT_COLUMNS = [
    'eqp_id_main_ovl',
    'ppid_main_ovl',
    'reticle_id_main_ovl',
    'eqp_id_base_ovl',
    'ppid_base_ovl',
    'reticle_id_base_ovl',
]


# ---------------------------------------------------------
# UFI event 정의
# ---------------------------------------------------------
# 기본값: UFI value > 0이면 해당 chip/item에서 event 발생.
# UFI value의 의미가 다른 경우 threshold와 비교 방식을 변경해야 한다.
UFI_EVENT_THRESHOLD = 0.0


# %%
# =========================================================
# 5단계 공통 함수
# =========================================================
def find_column_ci(df, candidates, required=True):
    """대소문자를 무시하고 첫 번째 존재 column을 반환한다."""
    lower_map = {str(column).lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        found = lower_map.get(str(candidate).lower())
        if found is not None:
            return found

    if required:
        raise KeyError(
            '다음 column 중 사용할 수 있는 column이 없습니다.\n'
            f'{candidates}\n\n현재 column:\n{df.columns.tolist()}'
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


def clean_numeric_series(series):
    return pd.to_numeric(series, errors='coerce')


def unique_clean_text_values(series):
    values = clean_text_series(series).dropna()
    values = [
        str(value).strip().upper()
        for value in values
        if str(value).strip()
    ]
    return sorted(set(values))


def summarize_context_values(data, context_column_map):
    """
    한 wafer/process/step/PPID fitting group의 context를 1개 record로 만든다.

    - 단일 값: 원래 값 유지
    - 여러 값: "|"로 모두 보존하고 AMBIGUOUS 표시
    - 일부 없음: PARTIAL
    - 전부 없음: NO_CONTEXT
    """
    output = {}
    ambiguous_columns = []
    present_count = 0

    for output_column, source_column in context_column_map.items():
        values = unique_clean_text_values(data[source_column])

        if not values:
            output[output_column] = pd.NA
            continue

        present_count += 1
        output[output_column] = '|'.join(values)

        if len(values) > 1:
            ambiguous_columns.append(output_column)

    if ambiguous_columns:
        status = 'AMBIGUOUS'
    elif present_count == len(context_column_map):
        status = 'MATCHED'
    elif present_count > 0:
        status = 'PARTIAL'
    else:
        status = 'NO_CONTEXT'

    # 3단계에서 이미 AMBIGUOUS/PARTIAL로 판단된 경우도 유지한다.
    if 'context_mapping_status' in data.columns:
        upstream_statuses = set(
            clean_text_series(data['context_mapping_status'])
            .dropna()
            .astype(str)
            .tolist()
        )

        if 'AMBIGUOUS' in upstream_statuses:
            status = 'AMBIGUOUS'
        elif status == 'MATCHED' and 'PARTIAL' in upstream_statuses:
            status = 'PARTIAL'

    output['context_mapping_status'] = status
    output['context_ambiguous_columns'] = '|'.join(ambiguous_columns)

    return output


def latest_file(directory, pattern):
    files = sorted(
        Path(directory).glob(pattern),
        key=lambda path: path.stat().st_mtime
    )

    if not files:
        return None

    return files[-1]


def safe_token(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def finite_max_abs(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        return np.nan

    return float(np.max(np.abs(array)))


def infer_regular_pitch(values):
    """
    chip center의 unique 좌표 차이에서 기본 pitch를 추정한다.
    누락 chip 때문에 2배, 3배 차이가 존재할 수 있으므로
    단순 median보다 작은 쪽의 반복 간격을 우선한다.
    """
    unique_values = np.sort(
        pd.Series(values)
        .dropna()
        .astype(float)
        .unique()
    )

    if len(unique_values) < 2:
        return np.nan

    diffs = np.diff(unique_values)
    diffs = diffs[diffs > 1e-10]

    if len(diffs) == 0:
        return np.nan

    rounded = np.round(diffs, 6)
    counts = pd.Series(rounded).value_counts()

    # 가장 자주 등장하는 차이를 사용한다.
    return float(counts.index[0])


def infer_layout_pitch_by_index(group, axis):
    """
    EDS chip index가 1만큼 다른 이웃 chip의 center 차이로 pitch를 계산한다.
    Staggered layout 등으로 전체 unique 좌표가 섞이는 경우를 줄인다.
    """
    if axis == 'x':
        index_column = '_chip_x_key'
        fixed_column = '_chip_y_key'
        center_column = '_chip_center_global_x'
    elif axis == 'y':
        index_column = '_chip_y_key'
        fixed_column = '_chip_x_key'
        center_column = '_chip_center_global_y'
    else:
        raise ValueError("axis는 'x' 또는 'y'여야 합니다.")

    work = group[[index_column, fixed_column, center_column]].copy()
    work['_numeric_index'] = pd.to_numeric(
        work[index_column],
        errors='coerce'
    )
    work[center_column] = pd.to_numeric(
        work[center_column],
        errors='coerce'
    )
    work = work.dropna(subset=['_numeric_index', fixed_column, center_column])

    neighbor_differences = []

    for _, fixed_group in work.groupby(fixed_column, dropna=False):
        fixed_group = (
            fixed_group
            .sort_values('_numeric_index')
            .drop_duplicates('_numeric_index')
        )

        index_diff = fixed_group['_numeric_index'].diff().to_numpy(float)
        center_diff = fixed_group[center_column].diff().abs().to_numpy(float)

        valid = (
            np.isfinite(index_diff)
            & np.isfinite(center_diff)
            & np.isclose(index_diff, 1.0)
            & (center_diff > 1e-10)
        )

        neighbor_differences.extend(center_diff[valid].tolist())

    if neighbor_differences:
        return float(np.median(neighbor_differences))

    return infer_regular_pitch(group[center_column])


def get_latest_stage_inputs():
    """메모리 변수를 우선하고, 없으면 최신 Pickle을 읽는다."""
    if (
        'ovl_filtered_all' in globals()
        and isinstance(ovl_filtered_all, pd.DataFrame)
        and not ovl_filtered_all.empty
    ):
        ovl_source = ovl_filtered_all.copy()
        print('현재 메모리의 ovl_filtered_all을 사용합니다.')

    else:
        ovl_pickle = latest_file(
            OUTPUT_DIR / 'OVL_MOD_FILTERED',
            'OVL_FILTERED_ALL__*.pkl'
        )

        if ovl_pickle is None:
            raise FileNotFoundError(
                'ovl_filtered_all 변수가 없고 OVL 통합 Pickle도 없습니다.'
            )

        ovl_source = pd.read_pickle(ovl_pickle)
        print(f'OVL Pickle을 사용합니다: {ovl_pickle}')

    if (
        'ufi_chip_filtered_all' in globals()
        and isinstance(ufi_chip_filtered_all, pd.DataFrame)
        and not ufi_chip_filtered_all.empty
    ):
        ufi_source = ufi_chip_filtered_all.copy()
        print('현재 메모리의 ufi_chip_filtered_all을 사용합니다.')

    else:
        ufi_pickle = latest_file(
            OUTPUT_DIR / 'UFI_CHIP_FILTERED' / RUN_TAG,
            'UFI_CHIP_FILTERED_ALL__*.pkl'
        )

        if ufi_pickle is None:
            raise FileNotFoundError(
                'ufi_chip_filtered_all 변수가 없고 UFI 통합 Pickle도 없습니다.'
            )

        ufi_source = pd.read_pickle(ufi_pickle)
        print(f'UFI Pickle을 사용합니다: {ufi_pickle}')

    return ovl_source, ufi_source


# %%
# =========================================================
# 안정화 4D Gaussian RBF
# =========================================================
class StableRBF4D:
    """
    P = [fcp_x, fcp_y, ftp_x, ftp_y]

    변경점:
      - 좌표별 표준화
      - k번째 최근접 거리 기반 auto sigma
      - (K + lambda I) regularization
      - Z 평균을 분리하여 멀리서 0 대신 평균으로 수렴
      - 최근접 4D 거리 반환
      - fitting diagnostics 저장
    """

    def __init__(
        self,
        sigma=None,
        lam=1e-3,
        neighbor_k=4,
        sigma_scale=1.5,
        condition_check_max_points=800,
    ):
        self.sigma_input = sigma
        self.sigma = None
        self.lam = float(lam)
        self.neighbor_k = int(neighbor_k)
        self.sigma_scale = float(sigma_scale)
        self.condition_check_max_points = int(condition_check_max_points)

        self.mean = None
        self.std = None
        self.P_ctrl = None
        self.w = None
        self.z_mean = None
        self.diagnostics = {}

    @staticmethod
    def _kernel(X, Y, sigma):
        d2 = cdist(X, Y, metric='sqeuclidean')
        return np.exp(-d2 / (2.0 * sigma ** 2))

    def _auto_sigma(self, P_scaled):
        n_points = len(P_scaled)

        if n_points < 2:
            raise ValueError('sigma 계산에 필요한 서로 다른 좌표가 없습니다.')

        distances = cdist(P_scaled, P_scaled)
        np.fill_diagonal(distances, np.inf)

        k = min(max(self.neighbor_k, 1), n_points - 1)
        kth_distance = np.partition(
            distances,
            kth=k - 1,
            axis=1
        )[:, k - 1]

        kth_distance = kth_distance[
            np.isfinite(kth_distance) & (kth_distance > 0)
        ]

        if len(kth_distance) == 0:
            raise ValueError('유효한 최근접 좌표 거리가 없습니다.')

        return float(np.median(kth_distance) * self.sigma_scale)

    @staticmethod
    def _aggregate_duplicate_points(P, Z):
        """동일 4D 좌표가 있으면 모델 학습용 Z만 평균 처리한다."""
        frame = pd.DataFrame(
            P,
            columns=['p0', 'p1', 'p2', 'p3']
        )

        z_columns = [f'z{index}' for index in range(Z.shape[1])]

        for index, column in enumerate(z_columns):
            frame[column] = Z[:, index]

        grouped = (
            frame
            .groupby(['p0', 'p1', 'p2', 'p3'], as_index=False, dropna=False)
            [z_columns]
            .mean()
        )

        P_unique = grouped[['p0', 'p1', 'p2', 'p3']].to_numpy(float)
        Z_unique = grouped[z_columns].to_numpy(float)

        return P_unique, Z_unique

    def fit(self, XY, IJ, Z):
        XY = np.asarray(XY, dtype=float)
        IJ = np.asarray(IJ, dtype=float)
        Z = np.asarray(Z, dtype=float)

        if Z.ndim == 1:
            Z = Z[:, None]

        if XY.ndim != 2 or IJ.ndim != 2:
            raise ValueError('XY와 IJ는 2차원 배열이어야 합니다.')

        if XY.shape[1] != 2 or IJ.shape[1] != 2:
            raise ValueError('XY와 IJ는 각각 두 개 column이어야 합니다.')

        if not (len(XY) == len(IJ) == len(Z)):
            raise ValueError('XY, IJ, Z의 row 수가 서로 다릅니다.')

        P = np.hstack([XY, IJ])
        valid = np.isfinite(P).all(axis=1) & np.isfinite(Z).all(axis=1)

        P = P[valid]
        Z = Z[valid]

        if len(P) < 8:
            raise ValueError(f'RBF fitting point가 너무 적습니다: {len(P)}')

        original_point_count = len(P)
        P, Z = self._aggregate_duplicate_points(P, Z)

        if len(P) < 8:
            raise ValueError(
                f'중복 좌표 정리 후 RBF point가 너무 적습니다: {len(P)}'
            )

        self.mean = P.mean(axis=0)
        self.std = P.std(axis=0)

        near_constant = self.std < 1e-8
        self.std = np.where(near_constant, 1.0, self.std)

        P_scaled = (P - self.mean) / self.std
        self.P_ctrl = P_scaled

        if self.sigma_input is None:
            self.sigma = self._auto_sigma(P_scaled)
        else:
            self.sigma = float(self.sigma_input)

        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError(f'잘못된 sigma: {self.sigma}')

        K = self._kernel(P_scaled, P_scaled, self.sigma)
        K_reg = K + self.lam * np.eye(len(K), dtype=float)

        self.z_mean = Z.mean(axis=0, keepdims=True)
        Z_centered = Z - self.z_mean

        try:
            self.w = np.linalg.solve(K_reg, Z_centered)
            solver = 'solve'
        except np.linalg.LinAlgError:
            self.w = np.linalg.pinv(K_reg, rcond=1e-10) @ Z_centered
            solver = 'pinv'

        fitted = K @ self.w + self.z_mean
        train_rmse = np.sqrt(np.mean((fitted - Z) ** 2, axis=0))

        if len(K_reg) <= self.condition_check_max_points:
            condition_number = float(np.linalg.cond(K_reg))
        else:
            condition_number = np.nan

        self.diagnostics = {
            'original_points': int(original_point_count),
            'unique_points': int(len(P)),
            'duplicate_points': int(original_point_count - len(P)),
            'sigma': float(self.sigma),
            'lambda': float(self.lam),
            'condition_number': condition_number,
            'max_abs_weight': float(np.max(np.abs(self.w))),
            'train_rmse_dx': float(train_rmse[0]),
            'train_rmse_dy': (
                float(train_rmse[1]) if len(train_rmse) >= 2 else np.nan
            ),
            'solver': solver,
        }

        return self

    def predict(self, XY, IJ, chunk_size=20000):
        if self.P_ctrl is None or self.w is None:
            raise RuntimeError('fit()을 먼저 실행해야 합니다.')

        XY = np.asarray(XY, dtype=float)
        IJ = np.asarray(IJ, dtype=float)
        P = np.hstack([XY, IJ])

        if P.shape[1] != 4:
            raise ValueError('predict 좌표는 총 4개 차원이어야 합니다.')

        P_scaled = (P - self.mean) / self.std

        predictions = []
        nearest_distances = []

        for start in range(0, len(P_scaled), chunk_size):
            stop = min(start + chunk_size, len(P_scaled))
            part = P_scaled[start:stop]

            distances = cdist(part, self.P_ctrl)
            nearest_distances.append(np.min(distances, axis=1))

            kernel = np.exp(
                -(distances ** 2) / (2.0 * self.sigma ** 2)
            )

            predictions.append(kernel @ self.w + self.z_mean)

        return (
            np.vstack(predictions),
            np.concatenate(nearest_distances)
        )


# %%
# =========================================================
# OVL/UFI 입력 정규화
# =========================================================
ovl_stage5_source, ufi_stage5_source = get_latest_stage_inputs()

ovl_work = ovl_stage5_source.copy()
ufi_work = ufi_stage5_source.copy()


# OVL column
OVL_ROOT_COL = find_column_ci(
    ovl_work,
    ['root_lot_id_key', 'root_lot_id', 'root_lot_id_ovl']
)
OVL_WAFER_COL = find_column_ci(
    ovl_work,
    ['wafer_id_key', 'wafer_id', 'wafer_id_ovl']
)
OVL_PROCESS_COL = find_column_ci(
    ovl_work,
    ['selected_process_id', 'process_id_ovl', 'process_id']
)
OVL_STEP_COL = find_column_ci(
    ovl_work,
    ['selected_step_seq', 'step_seq_ovl', 'step_seq']
)
OVL_PPID_COL = find_column_ci(
    ovl_work,
    ['selected_ppid_main_ovl', 'ppid_main_ovl', 'ppid_main']
)

FCP_X_COL = find_column_ci(ovl_work, ['fcp_x'])
FCP_Y_COL = find_column_ci(ovl_work, ['fcp_y'])
FTP_X_COL = find_column_ci(ovl_work, ['ftp_x'])
FTP_Y_COL = find_column_ci(ovl_work, ['ftp_y'])
DX_COL = find_column_ci(ovl_work, ['dx'])
DY_COL = find_column_ci(ovl_work, ['dy'])


# 3단계에서 전달된 context column을 정확히 확인한다.
# 누락된 상태로 5단계를 진행하면 마지막 결과에서 복구할 수 없으므로 중단한다.
OVL_CONTEXT_COLUMN_MAP = {
    context_column: find_column_ci(ovl_work, [context_column])
    for context_column in CONTEXT_COLUMNS
}

print('\n5단계에서 보존할 context column')
for output_column, source_column in OVL_CONTEXT_COLUMN_MAP.items():
    print(f'{output_column}: {source_column}')


# UFI column
UFI_ROOT_COL = find_column_ci(
    ufi_work,
    ['root_lot_id_key', 'root_lot_id', 'root_lot_id_ufi']
)
UFI_WAFER_COL = find_column_ci(
    ufi_work,
    ['wafer_id_key', 'wafer_id', 'wafer_id_ufi']
)
UFI_LINE_COL = find_column_ci(
    ufi_work,
    ['query_line_id', 'line_id']
)
UFI_PROCESS_COL = find_column_ci(
    ufi_work,
    ['query_process_id', 'process_id']
)
UFI_ITEM_COL = find_column_ci(
    ufi_work,
    ['item_id', 'query_item_id']
)
UFI_VALUE_COL = find_column_ci(
    ufi_work,
    ['value']
)
UFI_CHIP_X_COL = find_column_ci(
    ufi_work,
    ['chip_x_pos']
)
UFI_CHIP_Y_COL = find_column_ci(
    ufi_work,
    ['chip_y_pos']
)
UFI_TIME_COL = find_column_ci(
    ufi_work,
    ['tkout_time'],
    required=False
)


# 표준 key 생성
ovl_work['_root_key'] = clean_id_series(ovl_work[OVL_ROOT_COL], upper=True)
ovl_work['_wafer_key'] = clean_id_series(ovl_work[OVL_WAFER_COL])
ovl_work['_process_key'] = clean_text_series(ovl_work[OVL_PROCESS_COL])
ovl_work['_step_key'] = clean_text_series(ovl_work[OVL_STEP_COL])
ovl_work['_ppid_key'] = clean_text_series(ovl_work[OVL_PPID_COL])

for context_source_column in set(OVL_CONTEXT_COLUMN_MAP.values()):
    ovl_work[context_source_column] = clean_text_series(
        ovl_work[context_source_column]
    )

for column in [FCP_X_COL, FCP_Y_COL, FTP_X_COL, FTP_Y_COL, DX_COL, DY_COL]:
    ovl_work[column] = clean_numeric_series(ovl_work[column])

ufi_work['_root_key'] = clean_id_series(ufi_work[UFI_ROOT_COL], upper=True)
ufi_work['_wafer_key'] = clean_id_series(ufi_work[UFI_WAFER_COL])
ufi_work['_line_key'] = clean_text_series(ufi_work[UFI_LINE_COL])
ufi_work['_process_key'] = clean_text_series(ufi_work[UFI_PROCESS_COL])
ufi_work['_item_key'] = clean_text_series(ufi_work[UFI_ITEM_COL])
ufi_work['_chip_x_key'] = clean_id_series(ufi_work[UFI_CHIP_X_COL])
ufi_work['_chip_y_key'] = clean_id_series(ufi_work[UFI_CHIP_Y_COL])
ufi_work['_ufi_value'] = clean_numeric_series(ufi_work[UFI_VALUE_COL])

if UFI_TIME_COL is not None:
    ufi_work['_ufi_time'] = pd.to_datetime(
        ufi_work[UFI_TIME_COL],
        errors='coerce'
    )
else:
    ufi_work['_ufi_time'] = pd.NaT


# %%
# =========================================================
# fab.m_chip_prod_wf_layout 조회 및 UFI chip 위치 결합
# =========================================================
def query_chip_layout(line_id, process_id):
    params = {
        'table_name': 'fab.m_chip_prod_wf_layout',
        'line_id': line_id,
        'process_id': process_id,
        'setLimit': False,
    }

    print(f'[LAYOUT QUERY] {line_id} / {process_id}')
    data = bdq.getData(param=params)

    if data is None or len(data) == 0:
        raise ValueError(
            f'chip layout 조회 결과가 없습니다: {line_id}/{process_id}'
        )

    result = data.copy()
    result['_line_key'] = str(line_id).strip().upper()
    result['_process_key'] = str(process_id).strip().upper()

    return result


layout_frames = []

for (line_id, process_id), _ in ufi_work.groupby(
    ['_line_key', '_process_key'],
    dropna=False,
    sort=False
):
    if pd.isna(line_id) or pd.isna(process_id):
        continue

    layout_frames.append(query_chip_layout(line_id, process_id))

if not layout_frames:
    raise ValueError('조회된 chip layout이 없습니다.')

chip_layout_all = pd.concat(layout_frames, ignore_index=True)

LAYOUT_CHIP_X_COL = find_column_ci(
    chip_layout_all,
    ['eds_chip_x_pos', 'chip_x_pos']
)
LAYOUT_CHIP_Y_COL = find_column_ci(
    chip_layout_all,
    ['eds_chip_y_pos', 'chip_y_pos']
)
LAYOUT_CENTER_X_COL = find_column_ci(
    chip_layout_all,
    ['chip_center_x_pos', 'chip_center_x', 'center_x_pos']
)
LAYOUT_CENTER_Y_COL = find_column_ci(
    chip_layout_all,
    ['chip_center_y_pos', 'chip_center_y', 'center_y_pos']
)

chip_layout_all['_chip_x_key'] = clean_id_series(
    chip_layout_all[LAYOUT_CHIP_X_COL]
)
chip_layout_all['_chip_y_key'] = clean_id_series(
    chip_layout_all[LAYOUT_CHIP_Y_COL]
)
chip_layout_all['_chip_center_global_x'] = (
    clean_numeric_series(chip_layout_all[LAYOUT_CENTER_X_COL])
    * LAYOUT_POSITION_SCALE_TO_OVL
)
chip_layout_all['_chip_center_global_y'] = (
    clean_numeric_series(chip_layout_all[LAYOUT_CENTER_Y_COL])
    * LAYOUT_POSITION_SCALE_TO_OVL
)

layout_small = (
    chip_layout_all[
        [
            '_line_key',
            '_process_key',
            '_chip_x_key',
            '_chip_y_key',
            '_chip_center_global_x',
            '_chip_center_global_y',
        ]
    ]
    .dropna(
        subset=[
            '_chip_x_key',
            '_chip_y_key',
            '_chip_center_global_x',
            '_chip_center_global_y',
        ]
    )
    .drop_duplicates(
        subset=[
            '_line_key',
            '_process_key',
            '_chip_x_key',
            '_chip_y_key',
        ]
    )
)

ufi_with_layout = pd.merge(
    ufi_work,
    layout_small,
    on=[
        '_line_key',
        '_process_key',
        '_chip_x_key',
        '_chip_y_key',
    ],
    how='left',
    validate='many_to_one',
)

layout_match_rate = ufi_with_layout['_chip_center_global_x'].notna().mean()
print(f'Chip layout 결합률: {layout_match_rate:.2%}')

if layout_match_rate < 0.95:
    warnings.warn(
        f'Chip layout 결합률이 낮습니다: {layout_match_rate:.2%}. '
        'UFI와 layout의 chip_x_pos/y_pos 정의를 확인하세요.'
    )


# layout별 chip pitch 계산
layout_pitch_records = []

for (line_id, process_id), group in layout_small.groupby(
    ['_line_key', '_process_key'],
    dropna=False,
    sort=False
):
    pitch_x = infer_layout_pitch_by_index(group, axis='x')
    pitch_y = infer_layout_pitch_by_index(group, axis='y')

    layout_pitch_records.append({
        '_line_key': line_id,
        '_process_key': process_id,
        '_chip_pitch_x': pitch_x,
        '_chip_pitch_y': pitch_y,
    })

layout_pitch_df = pd.DataFrame(layout_pitch_records)

ufi_with_layout = pd.merge(
    ufi_with_layout,
    layout_pitch_df,
    on=['_line_key', '_process_key'],
    how='left',
    validate='many_to_one',
)

if ufi_with_layout[['_chip_pitch_x', '_chip_pitch_y']].isna().any().any():
    raise ValueError(
        'chip pitch를 계산하지 못한 line/process가 있습니다. '
        'chip layout의 center 좌표를 확인하세요.'
    )


# %%
# =========================================================
# Chip/item별 UFI event 집계
# =========================================================
ufi_with_layout['_ufi_event_row'] = (
    ufi_with_layout['_ufi_value'] > UFI_EVENT_THRESHOLD
)

ufi_chip_group_keys = [
    '_root_key',
    '_wafer_key',
    '_line_key',
    '_process_key',
    '_item_key',
    '_chip_x_key',
    '_chip_y_key',
    '_chip_center_global_x',
    '_chip_center_global_y',
    '_chip_pitch_x',
    '_chip_pitch_y',
]

ufi_chip_base = (
    ufi_with_layout
    .dropna(
        subset=[
            '_root_key',
            '_wafer_key',
            '_process_key',
            '_item_key',
            '_chip_x_key',
            '_chip_y_key',
            '_chip_center_global_x',
            '_chip_center_global_y',
        ]
    )
    .groupby(ufi_chip_group_keys, as_index=False, dropna=False)
    .agg(
        ufi_value_max=('_ufi_value', 'max'),
        ufi_value_mean=('_ufi_value', 'mean'),
        ufi_event=('_ufi_event_row', 'max'),
        ufi_record_count=('_ufi_event_row', 'size'),
        ufi_time_min=('_ufi_time', 'min'),
        ufi_time_max=('_ufi_time', 'max'),
    )
)

print('\nUFI event 분포')
print(ufi_chip_base['ufi_event'].value_counts(dropna=False))

if ufi_chip_base['ufi_event'].nunique(dropna=True) < 2:
    warnings.warn(
        'UFI event가 한 종류만 존재합니다. '
        'value > threshold 정의와 비불량 chip 포함 여부를 확인하세요.'
    )


# %%
# =========================================================
# 3x3 prediction point와 실측 point 생성 함수
# =========================================================
GRID_DEFINITION = [
    ('X0_Y0', -1, -1),
    ('X1_Y0',  0, -1),
    ('X2_Y0',  1, -1),
    ('X0_Y1', -1,  0),
    ('X1_Y1',  0,  0),
    ('X2_Y1',  1,  0),
    ('X0_Y2', -1,  1),
    ('X1_Y2',  0,  1),
    ('X2_Y2',  1,  1),
]


def get_shot_pitch(local_train, shot_centers):
    """
    1순위: _mod_ 안의 명시적 STEP/SHOT pitch
    2순위: 측정된 fcp center 차이에서 추정

    명시적 pitch가 없고 OVL sampling이 인접 shot을 하나도 포함하지 않으면
    추정 pitch가 실제 pitch의 정수배일 수 있으므로 model log를 확인해야 한다.
    """
    pitch_x_column = find_column_ci(
        local_train,
        [
            'STEP_PITCH_X',
            'step_pitch_x',
            'SHOT_PITCH_X',
            'shot_pitch_x',
        ],
        required=False,
    )
    pitch_y_column = find_column_ci(
        local_train,
        [
            'STEP_PITCH_Y',
            'step_pitch_y',
            'SHOT_PITCH_Y',
            'shot_pitch_y',
        ],
        required=False,
    )

    pitch_x = np.nan
    pitch_y = np.nan
    source_x = 'FCP_INFERRED'
    source_y = 'FCP_INFERRED'

    if pitch_x_column is not None:
        values = pd.to_numeric(
            local_train[pitch_x_column],
            errors='coerce'
        ).abs().dropna()

        if not values.empty:
            pitch_x = float(values.median() * SHOT_PITCH_SCALE_TO_OVL)
            source_x = pitch_x_column

    if pitch_y_column is not None:
        values = pd.to_numeric(
            local_train[pitch_y_column],
            errors='coerce'
        ).abs().dropna()

        if not values.empty:
            pitch_y = float(values.median() * SHOT_PITCH_SCALE_TO_OVL)
            source_y = pitch_y_column

    if not np.isfinite(pitch_x) or pitch_x <= 0:
        pitch_x = infer_regular_pitch(shot_centers['fcp_x'])

    if not np.isfinite(pitch_y) or pitch_y <= 0:
        pitch_y = infer_regular_pitch(shot_centers['fcp_y'])

    if not np.isfinite(pitch_x) or pitch_x <= 0:
        raise ValueError('shot pitch X를 계산하지 못했습니다.')

    if not np.isfinite(pitch_y) or pitch_y <= 0:
        raise ValueError('shot pitch Y를 계산하지 못했습니다.')

    return pitch_x, pitch_y, source_x, source_y


def assign_chip_shot_centers(chips, shot_centers, shot_pitch_x, shot_pitch_y):
    """
    측정된 shot만 최근접 배정하지 않고, 측정 fcp 한 점을 기준으로
    전체 exposure shot grid를 복원하여 각 chip의 shot center를 계산한다.
    """
    result = chips.copy()

    reference_fcp_x = float(shot_centers['fcp_x'].iloc[0])
    reference_fcp_y = float(shot_centers['fcp_y'].iloc[0])

    result['_assigned_fcp_x'] = (
        reference_fcp_x
        + np.round(
            (
                result['_chip_center_global_x']
                - reference_fcp_x
            )
            / shot_pitch_x
        )
        * shot_pitch_x
    )
    result['_assigned_fcp_y'] = (
        reference_fcp_y
        + np.round(
            (
                result['_chip_center_global_y']
                - reference_fcp_y
            )
            / shot_pitch_y
        )
        * shot_pitch_y
    )

    result['_chip_to_shot_center_distance'] = np.sqrt(
        (
            result['_chip_center_global_x']
            - result['_assigned_fcp_x']
        ) ** 2
        + (
            result['_chip_center_global_y']
            - result['_assigned_fcp_y']
        ) ** 2
    )

    result['_chip_center_ftp_x'] = (
        FTP_X_SIGN
        * (
            result['_chip_center_global_x']
            - result['_assigned_fcp_x']
        )
    )
    result['_chip_center_ftp_y'] = (
        FTP_Y_SIGN
        * (
            result['_chip_center_global_y']
            - result['_assigned_fcp_y']
        )
    )

    return result


def make_prediction_candidates(chips):
    records = []

    # underscore로 시작하는 column은 itertuples()에서 field명이 바뀔 수 있어
    # dict record로 순회한다.
    for chip_dict in chips.to_dict(orient='records'):

        offset_x_size = (
            float(chip_dict['_chip_pitch_x'])
            * CHIP_SAMPLE_EDGE_FRACTION
        )
        offset_y_size = (
            float(chip_dict['_chip_pitch_y'])
            * CHIP_SAMPLE_EDGE_FRACTION
        )

        for point_name, x_direction, y_direction in GRID_DEFINITION:
            ftp_x = (
                float(chip_dict['_chip_center_ftp_x'])
                + FTP_X_SIGN * x_direction * offset_x_size
            )
            ftp_y = (
                float(chip_dict['_chip_center_ftp_y'])
                + FTP_Y_SIGN * y_direction * offset_y_size
            )

            global_x = (
                float(chip_dict['_assigned_fcp_x'])
                + FTP_X_SIGN * ftp_x
            )
            global_y = (
                float(chip_dict['_assigned_fcp_y'])
                + FTP_Y_SIGN * ftp_y
            )

            records.append({
                '_root_key': chip_dict['_root_key'],
                '_wafer_key': chip_dict['_wafer_key'],
                '_line_key': chip_dict['_line_key'],
                '_process_key': chip_dict['_process_key'],
                '_chip_x_key': chip_dict['_chip_x_key'],
                '_chip_y_key': chip_dict['_chip_y_key'],
                'candidate_source': 'PREDICTED',
                'candidate_point_id': point_name,
                'candidate_fcp_x': float(chip_dict['_assigned_fcp_x']),
                'candidate_fcp_y': float(chip_dict['_assigned_fcp_y']),
                'candidate_ftp_x': ftp_x,
                'candidate_ftp_y': ftp_y,
                'candidate_global_x': global_x,
                'candidate_global_y': global_y,
                'shot_center_fcp_x': float(chip_dict['_assigned_fcp_x']),
                'shot_center_fcp_y': float(chip_dict['_assigned_fcp_y']),
                'shot_center_r_mm': float(chip_dict['_shot_center_r_mm']),
                'shot_region': chip_dict['_shot_region'],
            })

    return pd.DataFrame(records)


def make_measured_candidates(local_train, chips):
    if chips.empty or local_train.empty:
        return pd.DataFrame()

    chip_centers = chips[
        ['_chip_center_global_x', '_chip_center_global_y']
    ].to_numpy(float)
    tree = cKDTree(chip_centers)

    measured = local_train[
        [FCP_X_COL, FCP_Y_COL, FTP_X_COL, FTP_Y_COL, DX_COL, DY_COL]
    ].dropna().copy()

    if measured.empty:
        return pd.DataFrame()

    measured['_global_x'] = (
        measured[FCP_X_COL] + FTP_X_SIGN * measured[FTP_X_COL]
    )
    measured['_global_y'] = (
        measured[FCP_Y_COL] + FTP_Y_SIGN * measured[FTP_Y_COL]
    )

    measured_global = measured[['_global_x', '_global_y']].to_numpy(float)
    nearest_distance, nearest_index = tree.query(measured_global, k=1)

    nearest_chips = chips.iloc[nearest_index].reset_index(drop=True)
    measured = measured.reset_index(drop=True)

    delta_x = np.abs(
        measured['_global_x'].to_numpy(float)
        - nearest_chips['_chip_center_global_x'].to_numpy(float)
    )
    delta_y = np.abs(
        measured['_global_y'].to_numpy(float)
        - nearest_chips['_chip_center_global_y'].to_numpy(float)
    )

    inside = (
        delta_x
        <= (
            0.5
            * nearest_chips['_chip_pitch_x'].to_numpy(float)
            * ACTUAL_POINT_ASSIGN_TOLERANCE
        )
    ) & (
        delta_y
        <= (
            0.5
            * nearest_chips['_chip_pitch_y'].to_numpy(float)
            * ACTUAL_POINT_ASSIGN_TOLERANCE
        )
    )

    records = []

    for measured_index in np.flatnonzero(inside):
        point = measured.iloc[measured_index]
        chip = nearest_chips.iloc[measured_index]

        records.append({
            '_root_key': chip['_root_key'],
            '_wafer_key': chip['_wafer_key'],
            '_line_key': chip['_line_key'],
            '_process_key': chip['_process_key'],
            '_chip_x_key': chip['_chip_x_key'],
            '_chip_y_key': chip['_chip_y_key'],
            'candidate_source': 'MEASURED',
            'candidate_point_id': f'MEASURED_{measured_index:04d}',
            'candidate_fcp_x': float(point[FCP_X_COL]),
            'candidate_fcp_y': float(point[FCP_Y_COL]),
            'candidate_ftp_x': float(point[FTP_X_COL]),
            'candidate_ftp_y': float(point[FTP_Y_COL]),
            'candidate_global_x': float(point['_global_x']),
            'candidate_global_y': float(point['_global_y']),
            'shot_center_fcp_x': float(chip['_assigned_fcp_x']),
            'shot_center_fcp_y': float(chip['_assigned_fcp_y']),
            'shot_center_r_mm': float(chip['_shot_center_r_mm']),
            'shot_region': chip['_shot_region'],
            'predicted_dx': float(point[DX_COL]),
            'predicted_dy': float(point[DY_COL]),
            'nearest_train_distance_4d': 0.0,
            'candidate_valid': True,
            'candidate_qc_reason': 'MEASURED',
        })

    return pd.DataFrame(records)


# %%
# =========================================================
# Chip 대표값 집계 함수
# =========================================================
def add_selected_point_fields(output, prefix, row, value, signed_value=None):
    output[prefix] = float(value)

    if signed_value is not None:
        output[f'{prefix}_signed'] = float(signed_value)

    output[f'{prefix}_source'] = row['candidate_source']
    output[f'{prefix}_point_id'] = row['candidate_point_id']
    output[f'{prefix}_fcp_x'] = row['candidate_fcp_x']
    output[f'{prefix}_fcp_y'] = row['candidate_fcp_y']
    output[f'{prefix}_ftp_x'] = row['candidate_ftp_x']
    output[f'{prefix}_ftp_y'] = row['candidate_ftp_y']
    output[f'{prefix}_global_x'] = row['candidate_global_x']
    output[f'{prefix}_global_y'] = row['candidate_global_y']
    output[f'{prefix}_dx_at_point'] = row['predicted_dx']
    output[f'{prefix}_dy_at_point'] = row['predicted_dy']


def summarize_chip_candidates(group):
    valid = group[
        group['candidate_valid']
        & group['predicted_dx'].notna()
        & group['predicted_dy'].notna()
    ].copy()

    output = {
        'shot_center_fcp_x': float(group['shot_center_fcp_x'].iloc[0]),
        'shot_center_fcp_y': float(group['shot_center_fcp_y'].iloc[0]),
        'shot_center_r_mm': float(group['shot_center_r_mm'].iloc[0]),
        'shot_region': group['shot_region'].iloc[0],
        'candidate_count_total': int(len(group)),
        'candidate_count_valid': int(len(valid)),
        'predicted_count_total': int(
            group['candidate_source'].eq('PREDICTED').sum()
        ),
        'predicted_count_valid': int(
            (
                group['candidate_source'].eq('PREDICTED')
                & group['candidate_valid']
            ).sum()
        ),
        'measured_count': int(
            group['candidate_source'].eq('MEASURED').sum()
        ),
    }

    if valid.empty:
        output['representative_status'] = 'NO_VALID_CANDIDATE'
        return pd.Series(output)

    valid['radial_ovl'] = np.sqrt(
        valid['predicted_dx'] ** 2
        + valid['predicted_dy'] ** 2
    )

    dx_min_row = valid.loc[valid['predicted_dx'].idxmin()]
    dx_max_row = valid.loc[valid['predicted_dx'].idxmax()]
    dx_abs_row = valid.loc[valid['predicted_dx'].abs().idxmax()]

    dy_min_row = valid.loc[valid['predicted_dy'].idxmin()]
    dy_max_row = valid.loc[valid['predicted_dy'].idxmax()]
    dy_abs_row = valid.loc[valid['predicted_dy'].abs().idxmax()]

    radial_row = valid.loc[valid['radial_ovl'].idxmax()]

    add_selected_point_fields(
        output,
        'dx_min',
        dx_min_row,
        dx_min_row['predicted_dx']
    )
    add_selected_point_fields(
        output,
        'dx_max',
        dx_max_row,
        dx_max_row['predicted_dx']
    )
    add_selected_point_fields(
        output,
        'dx_absmax',
        dx_abs_row,
        abs(dx_abs_row['predicted_dx']),
        signed_value=dx_abs_row['predicted_dx']
    )

    add_selected_point_fields(
        output,
        'dy_min',
        dy_min_row,
        dy_min_row['predicted_dy']
    )
    add_selected_point_fields(
        output,
        'dy_max',
        dy_max_row,
        dy_max_row['predicted_dy']
    )
    add_selected_point_fields(
        output,
        'dy_absmax',
        dy_abs_row,
        abs(dy_abs_row['predicted_dy']),
        signed_value=dy_abs_row['predicted_dy']
    )

    add_selected_point_fields(
        output,
        'radial_ovl_max',
        radial_row,
        radial_row['radial_ovl']
    )

    output['representative_status'] = 'OK'

    return pd.Series(output)

