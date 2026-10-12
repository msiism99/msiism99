
# %%
# =========================================================
# 5단계 실행: wafer별 4D fitting -> chip 대표 OVL -> UFI 결합
# =========================================================
stage5_candidate_frames = []
stage5_representative_frames = []
stage5_model_logs = []

ovl_group_columns = [
    '_root_key',
    '_wafer_key',
    '_process_key',
    '_step_key',
    '_ppid_key',
]

ovl_groups = list(
    ovl_work.groupby(
        ovl_group_columns,
        dropna=False,
        sort=False
    )
)

for group_no, (group_key, local_train_raw) in enumerate(ovl_groups, start=1):
    (
        root_lot_id,
        wafer_id,
        process_id,
        step_seq,
        ppid_main,
    ) = group_key

    print('\n' + '=' * 90)
    print(
        f'[5단계 {group_no}/{len(ovl_groups)}] '
        f'{root_lot_id}/{wafer_id} / {process_id} / {step_seq}'
    )

    context_values = summarize_context_values(
        local_train_raw,
        OVL_CONTEXT_COLUMN_MAP,
    )

    print(
        f'context status: '
        f'{context_values["context_mapping_status"]}'
    )

    local_train = local_train_raw.dropna(
        subset=[
            FCP_X_COL,
            FCP_Y_COL,
            FTP_X_COL,
            FTP_Y_COL,
            DX_COL,
            DY_COL,
        ]
    ).copy()

    target_ufi = ufi_chip_base[
        ufi_chip_base['_root_key'].eq(root_lot_id)
        & ufi_chip_base['_wafer_key'].eq(wafer_id)
        & ufi_chip_base['_process_key'].eq(process_id)
    ].copy()

    if target_ufi.empty:
        stage5_model_logs.append({
            'root_lot_id': root_lot_id,
            'wafer_id': wafer_id,
            'process_id': process_id,
            'step_seq': step_seq,
            'ppid_main': ppid_main,
            'status': 'NO_UFI_CHIP',
            'train_points': len(local_train),
            'error': '',
            **context_values,
        })
        print('대상 UFI chip이 없어 건너뜁니다.')
        continue

    chips = (
        target_ufi[
            [
                '_root_key',
                '_wafer_key',
                '_line_key',
                '_process_key',
                '_chip_x_key',
                '_chip_y_key',
                '_chip_center_global_x',
                '_chip_center_global_y',
                '_chip_pitch_x',
                '_chip_pitch_y',
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    shot_centers = (
        local_train[[FCP_X_COL, FCP_Y_COL]]
        .drop_duplicates()
        .rename(columns={FCP_X_COL: 'fcp_x', FCP_Y_COL: 'fcp_y'})
        .reset_index(drop=True)
    )

    if len(local_train) < 8 or shot_centers.empty:
        stage5_model_logs.append({
            'root_lot_id': root_lot_id,
            'wafer_id': wafer_id,
            'process_id': process_id,
            'step_seq': step_seq,
            'ppid_main': ppid_main,
            'status': 'TOO_FEW_POINTS',
            'train_points': len(local_train),
            'error': '',
            **context_values,
        })
        print(f'fitting point 부족: {len(local_train)}')
        continue

    try:
        model = StableRBF4D(
            sigma=RBF_SIGMA,
            lam=RBF_LAMBDA,
            neighbor_k=RBF_SIGMA_NEIGHBOR_K,
            sigma_scale=RBF_SIGMA_SCALE,
            condition_check_max_points=RBF_CONDITION_CHECK_MAX_POINTS,
        )

        model.fit(
            local_train[[FCP_X_COL, FCP_Y_COL]].to_numpy(float),
            local_train[[FTP_X_COL, FTP_Y_COL]].to_numpy(float),
            local_train[[DX_COL, DY_COL]].to_numpy(float),
        )

        (
            shot_pitch_x,
            shot_pitch_y,
            shot_pitch_source_x,
            shot_pitch_source_y,
        ) = get_shot_pitch(local_train, shot_centers)

        chips_with_shot = assign_chip_shot_centers(
            chips,
            shot_centers,
            shot_pitch_x=shot_pitch_x,
            shot_pitch_y=shot_pitch_y,
        )
        chips_with_shot['_shot_center_r_mm'] = np.sqrt(
            chips_with_shot['_assigned_fcp_x'] ** 2
            + chips_with_shot['_assigned_fcp_y'] ** 2
        )
        chips_with_shot['_shot_region'] = np.where(
            chips_with_shot['_shot_center_r_mm'] > EDGE_SHOT_RADIUS_MM,
            'EDGE',
            'INNER',
        )
        predicted_candidates = make_prediction_candidates(chips_with_shot)

        prediction, nearest_distance = model.predict(
            predicted_candidates[
                ['candidate_fcp_x', 'candidate_fcp_y']
            ].to_numpy(float),
            predicted_candidates[
                ['candidate_ftp_x', 'candidate_ftp_y']
            ].to_numpy(float),
            chunk_size=RBF_PREDICT_CHUNK_SIZE,
        )

        predicted_candidates['predicted_dx'] = prediction[:, 0]
        predicted_candidates['predicted_dy'] = prediction[:, 1]
        predicted_candidates['nearest_train_distance_4d'] = nearest_distance

        dx_limit = max(
            PRED_LIMIT_FLOOR_NM,
            PRED_LIMIT_FACTOR * finite_max_abs(local_train[DX_COL])
        )
        dy_limit = max(
            PRED_LIMIT_FLOOR_NM,
            PRED_LIMIT_FACTOR * finite_max_abs(local_train[DY_COL])
        )

        support_limit = RBF_SUPPORT_SIGMA_MULTIPLIER * model.sigma

        finite_prediction = (
            np.isfinite(predicted_candidates['predicted_dx'])
            & np.isfinite(predicted_candidates['predicted_dy'])
        )
        inside_support = (
            predicted_candidates['nearest_train_distance_4d']
            <= support_limit
        )
        inside_value_limit = (
            predicted_candidates['predicted_dx'].abs() <= dx_limit
        ) & (
            predicted_candidates['predicted_dy'].abs() <= dy_limit
        )

        predicted_candidates['candidate_valid'] = (
            finite_prediction & inside_support & inside_value_limit
        )

        predicted_candidates['candidate_qc_reason'] = np.select(
            [
                ~finite_prediction,
                ~inside_support,
                ~inside_value_limit,
            ],
            [
                'NON_FINITE',
                'OUTSIDE_4D_SUPPORT',
                'PREDICTION_OVERSHOOT',
            ],
            default='OK'
        )

        measured_candidates = make_measured_candidates(
            local_train=local_train,
            chips=chips_with_shot,
        )

        candidates = pd.concat(
            [predicted_candidates, measured_candidates],
            ignore_index=True,
            sort=False,
        )

        candidates['selected_step_seq'] = step_seq
        candidates['selected_ppid_main_ovl'] = ppid_main
        candidates['selected_process_id'] = process_id
        candidates['root_lot_id'] = root_lot_id
        candidates['wafer_id'] = wafer_id
        candidates['rlid_wid'] = f'{root_lot_id}_{wafer_id}'

        for context_column, context_value in context_values.items():
            candidates[context_column] = context_value

        candidate_group_keys = [
            '_root_key',
            '_wafer_key',
            '_line_key',
            '_process_key',
            '_chip_x_key',
            '_chip_y_key',
        ]

        representatives = (
            candidates
            .groupby(candidate_group_keys, dropna=False, sort=False)
            .apply(summarize_chip_candidates)
            .reset_index()
        )

        representatives['selected_process_id'] = process_id
        representatives['selected_step_seq'] = step_seq
        representatives['selected_ppid_main_ovl'] = ppid_main
        representatives['root_lot_id'] = root_lot_id
        representatives['wafer_id'] = wafer_id
        representatives['rlid_wid'] = f'{root_lot_id}_{wafer_id}'

        for context_column, context_value in context_values.items():
            representatives[context_column] = context_value

        # 방향별 curve용 값. dx_min이 +값인 chip은 음의 방향 데이터가 아니며,
        # dx_max가 -값인 chip은 양의 방향 데이터가 아니다.
        representatives['dx_negative'] = representatives['dx_min'].where(
            representatives['dx_min'] < 0
        )
        representatives['dx_positive'] = representatives['dx_max'].where(
            representatives['dx_max'] > 0
        )
        representatives['dy_negative'] = representatives['dy_min'].where(
            representatives['dy_min'] < 0
        )
        representatives['dy_positive'] = representatives['dy_max'].where(
            representatives['dy_max'] > 0
        )

        stage5_candidate_frames.append(candidates)
        stage5_representative_frames.append(representatives)

        log_record = {
            'root_lot_id': root_lot_id,
            'wafer_id': wafer_id,
            'process_id': process_id,
            'step_seq': step_seq,
            'ppid_main': ppid_main,
            'status': 'OK',
            'train_points': len(local_train),
            'shot_count': len(shot_centers),
            'ufi_chip_count': len(chips),
            'inner_chip_count': int(
                chips_with_shot['_shot_region'].eq('INNER').sum()
            ),
            'edge_chip_count': int(
                chips_with_shot['_shot_region'].eq('EDGE').sum()
            ),
            'edge_shot_radius_mm': EDGE_SHOT_RADIUS_MM,
            'prediction_candidates': len(predicted_candidates),
            'valid_predictions': int(
                predicted_candidates['candidate_valid'].sum()
            ),
            'measured_candidates': len(measured_candidates),
            'dx_prediction_limit_nm': dx_limit,
            'dy_prediction_limit_nm': dy_limit,
            'support_limit_4d': support_limit,
            'shot_pitch_x': shot_pitch_x,
            'shot_pitch_y': shot_pitch_y,
            'shot_pitch_source_x': shot_pitch_source_x,
            'shot_pitch_source_y': shot_pitch_source_y,
            'error': '',
            **context_values,
            **model.diagnostics,
        }
        stage5_model_logs.append(log_record)

        print(
            f"sigma={model.sigma:.4g}, lambda={model.lam:g}, "
            f"valid prediction="
            f"{int(predicted_candidates['candidate_valid'].sum()):,}/"
            f"{len(predicted_candidates):,}, "
            f"measured candidates={len(measured_candidates):,}"
        )

    except Exception as error:
        stage5_model_logs.append({
            'root_lot_id': root_lot_id,
            'wafer_id': wafer_id,
            'process_id': process_id,
            'step_seq': step_seq,
            'ppid_main': ppid_main,
            'status': 'FAILED',
            'train_points': len(local_train),
            'error': str(error),
            **context_values,
        })
        print(f'처리 실패: {error}')


df_stage5_model_log = pd.DataFrame(stage5_model_logs)

if not stage5_representative_frames:
    failed_log_path = STAGE5_OUTPUT_DIR / 'STAGE5_FAILED_LOG.xlsx'
    df_stage5_model_log.to_excel(failed_log_path, index=False)
    raise ValueError(
        '생성된 chip 대표 OVL이 없습니다.\n'
        f'로그: {failed_log_path}'
    )

ovl_chip_representatives = pd.concat(
    stage5_representative_frames,
    ignore_index=True,
)

ovl_chip_sample_points = pd.concat(
    stage5_candidate_frames,
    ignore_index=True,
)


# UFI item별 집계와 OVL chip 대표값 결합
merge_keys = [
    '_root_key',
    '_wafer_key',
    '_line_key',
    '_process_key',
    '_chip_x_key',
    '_chip_y_key',
]

ovl_ufi_chip_all = pd.merge(
    ovl_chip_representatives,
    ufi_chip_base,
    on=merge_keys,
    how='inner',
    # 동일 chip에 여러 OVL step/PPID 또는 여러 UFI item이 존재할 수 있다.
    validate='many_to_many',
)

ovl_ufi_chip_all = ovl_ufi_chip_all.rename(columns={
    '_line_key': 'line_id',
    '_process_key': 'process_id',
    '_item_key': 'ufi_item_id',
    '_chip_x_key': 'chip_x_pos',
    '_chip_y_key': 'chip_y_pos',
})

ovl_ufi_chip_all['ufi_event'] = ovl_ufi_chip_all['ufi_event'].astype(bool)


# context가 최종 chip-level OVL-UFI 데이터까지 도달했는지 즉시 검증한다.
missing_result_context_columns = [
    column
    for column in CONTEXT_COLUMNS
    if column not in ovl_ufi_chip_all.columns
]

if missing_result_context_columns:
    raise KeyError(
        '5단계 최종 데이터에 context column이 없습니다. '
        '수정된 3단계를 먼저 실행하세요.\n'
        f'{missing_result_context_columns}'
    )

stage5_context_summary = (
    ovl_ufi_chip_all
    .groupby(
        CONTEXT_COLUMNS + ['context_mapping_status'],
        dropna=False,
        as_index=False,
    )
    .agg(
        chip_rows=('ufi_event', 'size'),
        event_chips=('ufi_event', 'sum'),
        ufi_rate=('ufi_event', 'mean'),
        wafers=('rlid_wid', 'nunique'),
    )
)


# %%
# =========================================================
# 5단계 저장
# =========================================================
stage5_timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

STAGE5_RESULT_PICKLE = STAGE5_OUTPUT_DIR / (
    f'OVL_UFI_CHIP_ALL__{stage5_timestamp}.pkl'
)
STAGE5_REPRESENTATIVE_PICKLE = STAGE5_OUTPUT_DIR / (
    f'OVL_CHIP_REPRESENTATIVES__{stage5_timestamp}.pkl'
)
STAGE5_SAMPLE_PICKLE = STAGE5_OUTPUT_DIR / (
    f'OVL_CHIP_SAMPLE_POINTS__{stage5_timestamp}.pkl'
)
STAGE5_LOG_EXCEL = STAGE5_OUTPUT_DIR / (
    f'STAGE5_MODEL_LOG__{stage5_timestamp}.xlsx'
)

ovl_ufi_chip_all.to_pickle(STAGE5_RESULT_PICKLE)
ovl_chip_representatives.to_pickle(STAGE5_REPRESENTATIVE_PICKLE)

if SAVE_STAGE5_SAMPLE_POINTS:
    ovl_chip_sample_points.to_pickle(STAGE5_SAMPLE_PICKLE)

with pd.ExcelWriter(STAGE5_LOG_EXCEL, engine='openpyxl') as writer:
    df_stage5_model_log.to_excel(
        writer,
        sheet_name='MODEL_LOG',
        index=False
    )

    stage5_context_summary.to_excel(
        writer,
        sheet_name='CONTEXT_SUMMARY',
        index=False,
    )

    (
        ovl_ufi_chip_all
        .groupby(
            [
                'selected_process_id',
                'selected_step_seq',
                'ufi_item_id',
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            chip_rows=('ufi_event', 'size'),
            event_chips=('ufi_event', 'sum'),
            ufi_rate=('ufi_event', 'mean'),
            wafers=('rlid_wid', 'nunique'),
        )
        .to_excel(
            writer,
            sheet_name='MERGE_SUMMARY',
            index=False
        )
    )

print('\n' + '=' * 90)
print('5단계 완료')
print(f'Chip 대표 OVL: {len(ovl_chip_representatives):,} rows')
print(f'OVL-UFI 결합: {len(ovl_ufi_chip_all):,} rows')
print(f'결과 Pickle: {STAGE5_RESULT_PICKLE}')
print(f'모델 로그: {STAGE5_LOG_EXCEL}')


# %% [markdown]
# # 6단계: 0.1 nm bin별 UFI rate 사발커브


# %%
# =========================================================
# 6단계 사용자 설정
# =========================================================
STAGE6_OUTPUT_DIR = OUTPUT_DIR / 'UFI_BOWL_CURVES' / RUN_TAG
STAGE6_CHART_DIR = STAGE6_OUTPUT_DIR / 'CHARTS'

STAGE6_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE6_CHART_DIR.mkdir(parents=True, exist_ok=True)

OVL_BIN_WIDTH_NM = 0.1

# 전체 bin 결과는 모두 저장하고, 그래프에는 이 개수 이상인 bin만 표시한다.
MIN_BIN_CHIP_COUNT_FOR_PLOT = 20

# 그래프는 DX, DY, RADIAL을 각각 별도 파일로 저장한다.
CURVE_AXES = ['DX', 'DY', 'RADIAL']

# 현재처럼 전부 합친 결과와 INNER/EDGE shot을 각각 생성한다.
SHOT_SCOPES = ['ALL', 'INNER', 'EDGE']

# 기존 6단계 동작을 유지한다.
# 새 context column(eqp/reticle/base)은 의도적으로 여기에 추가하지 않는다.
# 따라서 기존 사발커브는 context를 새로 세분화하지 않고 동일하게 생성된다.
# PPID가 여러 개인 경우를 분리하는 기존 기준도 그대로 유지한다.
CURVE_GROUP_COLUMNS = [
    'selected_process_id',
    'selected_step_seq',
    'selected_ppid_main_ovl',
    'ufi_item_id',
]


# %%
# =========================================================
# 6단계 함수
# =========================================================
def make_ovl_bins(values, width):
    numeric = pd.to_numeric(values, errors='coerce').to_numpy(float)
    bin_left = np.floor((numeric + 1e-12) / width) * width
    bin_left = np.round(bin_left, 10)
    bin_right = np.round(bin_left + width, 10)
    bin_center = np.round(bin_left + width / 2.0, 10)

    return bin_left, bin_right, bin_center


def binomial_rate_ci(event_count, total_count, z=1.96):
    """Wilson 95% CI. 반환 단위는 percent."""
    if total_count <= 0:
        return np.nan, np.nan

    p = event_count / total_count
    denominator = 1.0 + z ** 2 / total_count
    center = (
        p + z ** 2 / (2.0 * total_count)
    ) / denominator
    half_width = (
        z
        * math.sqrt(
            p * (1.0 - p) / total_count
            + z ** 2 / (4.0 * total_count ** 2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width) * 100.0,
        min(1.0, center + half_width) * 100.0,
    )


def make_axis_observations(data, curve_axis):
    """
    DX/DY는 음의 방향과 양의 방향을 별도로 구성한다.

    - NEGATIVE: min 값이 실제로 0보다 작은 chip만 사용
    - POSITIVE: max 값이 실제로 0보다 큰 chip만 사용

    따라서 dx_min=+2인 chip은 NEGATIVE branch에 들어가지 않고,
    해당 chip의 dx_max가 POSITIVE branch에 들어간다.
    """
    if curve_axis == 'DX':
        negative_metric = 'dx_negative'
        positive_metric = 'dx_positive'
    elif curve_axis == 'DY':
        negative_metric = 'dy_negative'
        positive_metric = 'dy_positive'
    elif curve_axis == 'RADIAL':
        radial = data[['radial_ovl_max', 'ufi_event']].copy()
        radial['ovl_value'] = pd.to_numeric(
            radial['radial_ovl_max'],
            errors='coerce'
        )
        radial = radial[
            radial['ovl_value'].ge(0)
        ].dropna(subset=['ovl_value', 'ufi_event'])
        radial['direction'] = 'RADIAL'
        radial['source_metric'] = 'radial_ovl_max'

        return radial[['ovl_value', 'ufi_event', 'direction', 'source_metric']]
    else:
        raise ValueError(f'지원하지 않는 curve axis: {curve_axis}')

    negative = data[[negative_metric, 'ufi_event']].copy()
    negative['ovl_value'] = pd.to_numeric(
        negative[negative_metric],
        errors='coerce'
    )
    negative = negative[
        negative['ovl_value'].lt(0)
    ].dropna(subset=['ovl_value', 'ufi_event'])
    negative['direction'] = 'NEGATIVE'
    negative['source_metric'] = negative_metric

    positive = data[[positive_metric, 'ufi_event']].copy()
    positive['ovl_value'] = pd.to_numeric(
        positive[positive_metric],
        errors='coerce'
    )
    positive = positive[
        positive['ovl_value'].gt(0)
    ].dropna(subset=['ovl_value', 'ufi_event'])
    positive['direction'] = 'POSITIVE'
    positive['source_metric'] = positive_metric

    return pd.concat(
        [
            negative[['ovl_value', 'ufi_event', 'direction', 'source_metric']],
            positive[['ovl_value', 'ufi_event', 'direction', 'source_metric']],
        ],
        ignore_index=True,
    )


def build_binned_curve(
    observations,
    curve_axis,
    shot_scope,
    group_values,
):
    work = observations.copy()

    if work.empty:
        return pd.DataFrame()

    (
        work['ovl_bin_left'],
        work['ovl_bin_right'],
        work['ovl_bin_center'],
    ) = make_ovl_bins(work['ovl_value'], OVL_BIN_WIDTH_NM)

    binned = (
        work
        .groupby(
            [
                'direction',
                'source_metric',
                'ovl_bin_left',
                'ovl_bin_right',
                'ovl_bin_center',
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            chip_count=('ufi_event', 'size'),
            ufi_event_count=('ufi_event', 'sum'),
            ovl_value_mean=('ovl_value', 'mean'),
            ovl_value_min=('ovl_value', 'min'),
            ovl_value_max=('ovl_value', 'max'),
        )
    )

    binned['ufi_rate_pct'] = (
        binned['ufi_event_count']
        / binned['chip_count']
        * 100.0
    )

    confidence = binned.apply(
        lambda row: binomial_rate_ci(
            int(row['ufi_event_count']),
            int(row['chip_count'])
        ),
        axis=1,
        result_type='expand',
    )
    confidence.columns = ['ufi_rate_ci_low_pct', 'ufi_rate_ci_high_pct']
    binned = pd.concat([binned, confidence], axis=1)

    binned['curve_axis'] = curve_axis
    binned['shot_scope'] = shot_scope
    binned['bin_width_nm'] = OVL_BIN_WIDTH_NM

    for column, value in group_values.items():
        binned[column] = value

    return binned


def plot_axis_curve(
    axis_curve_data,
    curve_axis,
    shot_scope,
    group_values,
    output_path,
):
    fig, axis = plt.subplots(figsize=(10.5, 6.5))

    color_map = {
        'NEGATIVE': '#1565C0',
        'POSITIVE': '#C62828',
        'RADIAL': '#2E7D32',
    }
    fill_color_map = {
        'NEGATIVE': '#64B5F6',
        'POSITIVE': '#EF9A9A',
        'RADIAL': '#81C784',
    }

    plotted = False

    for direction in ['NEGATIVE', 'POSITIVE', 'RADIAL']:
        plot_data = axis_curve_data[
            axis_curve_data['direction'].eq(direction)
            & (
                axis_curve_data['chip_count']
                >= MIN_BIN_CHIP_COUNT_FOR_PLOT
            )
        ].sort_values('ovl_bin_center')

        if plot_data.empty:
            continue

        plotted = True
        x = plot_data['ovl_bin_center'].to_numpy(float)
        y = plot_data['ufi_rate_pct'].to_numpy(float)
        ci_low = plot_data['ufi_rate_ci_low_pct'].to_numpy(float)
        ci_high = plot_data['ufi_rate_ci_high_pct'].to_numpy(float)

        axis.plot(
            x,
            y,
            marker='o',
            markersize=5,
            linewidth=1.8,
            color=color_map[direction],
            label=direction.title(),
        )
        axis.fill_between(
            x,
            ci_low,
            ci_high,
            color=fill_color_map[direction],
            alpha=0.22,
            linewidth=0,
        )

        for row in plot_data.itertuples(index=False):
            axis.annotate(
                f'n={int(row.chip_count)}',
                (row.ovl_bin_center, row.ufi_rate_pct),
                xytext=(0, 6),
                textcoords='offset points',
                ha='center',
                fontsize=7,
                alpha=0.72,
            )

    if not plotted:
        axis.text(
            0.5,
            0.5,
            f'No bin with n >= {MIN_BIN_CHIP_COUNT_FOR_PLOT}',
            ha='center',
            va='center',
            transform=axis.transAxes,
        )

    axis.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    axis.set_ylim(bottom=0)
    axis.set_xlabel(
        f'{curve_axis} OVL ({OVL_BIN_WIDTH_NM:.1f} nm bin center)'
    )
    axis.set_ylabel('UFI rate (%)')
    axis.grid(alpha=0.25)

    if plotted:
        axis.legend(frameon=False)

    group_title = ' / '.join(
        f'{column}={value}'
        for column, value in group_values.items()
    )
    axis.set_title(
        f'{curve_axis} UFI curve — {shot_scope}\n{group_title}',
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# %%
# =========================================================
# 6단계 실행
# =========================================================
if (
    'ovl_ufi_chip_all' not in globals()
    or not isinstance(ovl_ufi_chip_all, pd.DataFrame)
    or ovl_ufi_chip_all.empty
):
    latest_stage5_result = latest_file(
        STAGE5_OUTPUT_DIR,
        'OVL_UFI_CHIP_ALL__*.pkl'
    )

    if latest_stage5_result is None:
        raise FileNotFoundError(
            'ovl_ufi_chip_all 변수가 없고 5단계 결과 Pickle도 없습니다.'
        )

    ovl_ufi_chip_all = pd.read_pickle(latest_stage5_result)
    print(f'5단계 Pickle을 사용합니다: {latest_stage5_result}')


required_curve_columns = [
    'dx_negative',
    'dx_positive',
    'dy_negative',
    'dy_positive',
    'radial_ovl_max',
    'shot_region',
    'ufi_event',
]

missing_curve_columns = [
    column
    for column in required_curve_columns
    if column not in ovl_ufi_chip_all.columns
]

if missing_curve_columns:
    raise KeyError(
        '6단계에 필요한 column이 없습니다. 수정된 5단계를 먼저 실행하세요.\n'
        f'{missing_curve_columns}'
    )


curve_frames = []

curve_groups = list(
    ovl_ufi_chip_all.groupby(
        CURVE_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    )
)

for group_no, (group_key, group_data) in enumerate(curve_groups, start=1):
    if not isinstance(group_key, tuple):
        group_key = (group_key,)

    group_values = dict(zip(CURVE_GROUP_COLUMNS, group_key))

    print('\n' + '-' * 90)
    print(f'[6단계 {group_no}/{len(curve_groups)}] {group_values}')

    file_token = '__'.join(
        safe_token(group_values[column])
        for column in CURVE_GROUP_COLUMNS
    )

    for shot_scope in SHOT_SCOPES:
        if shot_scope == 'ALL':
            scope_data = group_data.copy()
        else:
            scope_data = group_data[
                group_data['shot_region'].eq(shot_scope)
            ].copy()

        if scope_data.empty:
            print(f'  {shot_scope}: 대상 chip 없음')
            continue

        scope_chart_dir = STAGE6_CHART_DIR / shot_scope
        scope_chart_dir.mkdir(parents=True, exist_ok=True)

        for curve_axis in CURVE_AXES:
            observations = make_axis_observations(
                data=scope_data,
                curve_axis=curve_axis,
            )

            binned = build_binned_curve(
                observations=observations,
                curve_axis=curve_axis,
                shot_scope=shot_scope,
                group_values=group_values,
            )

            if binned.empty:
                print(f'  {shot_scope}/{curve_axis}: 유효 데이터 없음')
                continue

            curve_frames.append(binned)

            chart_path = scope_chart_dir / (
                f'UFI_BOWL__{file_token}__{curve_axis}.png'
            )

            plot_axis_curve(
                axis_curve_data=binned,
                curve_axis=curve_axis,
                shot_scope=shot_scope,
                group_values=group_values,
                output_path=chart_path,
            )


if not curve_frames:
    raise ValueError('생성된 UFI rate bin 결과가 없습니다.')

ufi_bowl_curve_bins = pd.concat(curve_frames, ignore_index=True)


# %%
# =========================================================
# 6단계 저장
# =========================================================
stage6_timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

STAGE6_EXCEL_PATH = STAGE6_OUTPUT_DIR / (
    f'UFI_BOWL_CURVES__{stage6_timestamp}.xlsx'
)
STAGE6_PICKLE_PATH = STAGE6_OUTPUT_DIR / (
    f'UFI_BOWL_CURVES__{stage6_timestamp}.pkl'
)

ufi_bowl_curve_bins.to_pickle(STAGE6_PICKLE_PATH)

curve_group_summary_frames = []

for shot_scope in SHOT_SCOPES:
    if shot_scope == 'ALL':
        summary_source = ovl_ufi_chip_all.copy()
    else:
        summary_source = ovl_ufi_chip_all[
            ovl_ufi_chip_all['shot_region'].eq(shot_scope)
        ].copy()

    if summary_source.empty:
        continue

    summary = (
        summary_source
        .groupby(CURVE_GROUP_COLUMNS, dropna=False, as_index=False)
        .agg(
            chip_count=('ufi_event', 'size'),
            ufi_event_count=('ufi_event', 'sum'),
            ufi_rate_pct=(
                'ufi_event',
                lambda values: values.mean() * 100.0
            ),
            wafer_count=('rlid_wid', 'nunique'),
        )
    )
    summary['shot_scope'] = shot_scope
    curve_group_summary_frames.append(summary)

curve_group_summary = pd.concat(
    curve_group_summary_frames,
    ignore_index=True,
)

with pd.ExcelWriter(STAGE6_EXCEL_PATH, engine='openpyxl') as writer:
    ufi_bowl_curve_bins.to_excel(
        writer,
        sheet_name='BINNED_UFI_RATE',
        index=False,
    )
    curve_group_summary.to_excel(
        writer,
        sheet_name='GROUP_SUMMARY',
        index=False,
    )

print('\n' + '=' * 90)
print('6단계 완료')
print(f'Binned curve rows: {len(ufi_bowl_curve_bins):,}')
print(f'Curve Excel: {STAGE6_EXCEL_PATH}')
print(f'Curve Pickle: {STAGE6_PICKLE_PATH}')
print(f'Chart directory: {STAGE6_CHART_DIR}')
