from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperMetricTriple:
    mae: float
    max_ae: float
    mse: float


@dataclass(frozen=True)
class PaperStepReference:
    paper_model: PaperMetricTriple
    numerical: PaperMetricTriple


@dataclass(frozen=True)
class PaperScenarioReference:
    table_id: str
    grid_label: str
    sample_count: int
    steps: dict[int, PaperStepReference]


PAPER_UNIFORM_REFERENCES: dict[str, PaperScenarioReference] = {
    "smooth_256": PaperScenarioReference(
        table_id="Table 3",
        grid_label="107x107",
        sample_count=528,
        steps={
            5: PaperStepReference(
                paper_model=PaperMetricTriple(1.138546e-03, 8.424489e-03, 2.568785e-06),
                numerical=PaperMetricTriple(2.599575e-03, 1.020075e-02, 1.142227e-05),
            ),
            10: PaperStepReference(
                paper_model=PaperMetricTriple(5.828545e-04, 4.830513e-03, 7.112521e-07),
                numerical=PaperMetricTriple(1.290843e-03, 1.427928e-02, 4.454964e-06),
            ),
            20: PaperStepReference(
                paper_model=PaperMetricTriple(4.759801e-04, 4.287496e-03, 5.615322e-07),
                numerical=PaperMetricTriple(1.089934e-03, 1.381520e-02, 4.083200e-06),
            ),
        },
    ),
    "smooth_266": PaperScenarioReference(
        table_id="Table 5",
        grid_label="111x111",
        sample_count=552,
        steps={
            5: PaperStepReference(
                paper_model=PaperMetricTriple(1.105992e-03, 7.895379e-03, 2.557006e-06),
                numerical=PaperMetricTriple(2.359816e-03, 1.374033e-02, 9.587425e-06),
            ),
            10: PaperStepReference(
                paper_model=PaperMetricTriple(5.832588e-04, 6.367638e-03, 8.334263e-07),
                numerical=PaperMetricTriple(1.243609e-03, 1.146487e-02, 3.884756e-06),
            ),
            20: PaperStepReference(
                paper_model=PaperMetricTriple(4.649859e-04, 6.305406e-03, 6.142720e-07),
                numerical=PaperMetricTriple(1.035165e-03, 1.220033e-02, 3.608487e-06),
            ),
        },
    ),
    "smooth_276": PaperScenarioReference(
        table_id="Table 7",
        grid_label="114x114",
        sample_count=564,
        steps={
            5: PaperStepReference(
                paper_model=PaperMetricTriple(1.164241e-03, 6.791887e-03, 2.423722e-06),
                numerical=PaperMetricTriple(2.301648e-03, 1.481315e-02, 9.299462e-06),
            ),
            10: PaperStepReference(
                paper_model=PaperMetricTriple(6.946607e-04, 4.271347e-03, 8.500361e-07),
                numerical=PaperMetricTriple(1.200980e-03, 1.252291e-02, 3.936283e-06),
            ),
            20: PaperStepReference(
                paper_model=PaperMetricTriple(6.165151e-04, 4.365713e-03, 7.043411e-07),
                numerical=PaperMetricTriple(9.802761e-04, 1.311203e-02, 3.633351e-06),
            ),
        },
    ),
    "acute_276": PaperScenarioReference(
        table_id="Table 8",
        grid_label="129x129",
        sample_count=672,
        steps={
            5: PaperStepReference(
                paper_model=PaperMetricTriple(2.815830e-03, 3.444712e-02, 2.717994e-05),
                numerical=PaperMetricTriple(4.598858e-03, 8.030253e-02, 6.706189e-05),
            ),
            10: PaperStepReference(
                paper_model=PaperMetricTriple(1.196355e-03, 3.333340e-02, 1.127559e-05),
                numerical=PaperMetricTriple(2.178264e-03, 8.390227e-02, 4.979660e-05),
            ),
            20: PaperStepReference(
                paper_model=PaperMetricTriple(1.032981e-03, 3.302984e-02, 1.105503e-05),
                numerical=PaperMetricTriple(1.716480e-03, 8.419102e-02, 5.001022e-05),
            ),
        },
    ),
}


def get_paper_reference(exp_id: str, step: int) -> PaperStepReference:
    scenario = PAPER_UNIFORM_REFERENCES[exp_id]
    return scenario.steps[int(step)]
