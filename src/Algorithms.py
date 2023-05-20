import os
from typing import Callable, Literal
import torch
import src.models as models
from src.utils.constants import PLACEMENT, R_RECEIVERS, SAMPLING_FREQUENCY, TRAINED_MODELS_PATH
from src.utils.utils import max_tau, merge_signal_tdoa, split_signal_tdoa
from src.utils.types import DatasetInfo


class Algorithms():
    def __init__(
        self,
        device: torch.device,
        fine: bool = False,
        MLAT: Literal['PSO', 'LS']  = 'PSO'
    ):
        self.device = device

        load_model = lambda model_name: torch.load(os.path.join(
            TRAINED_MODELS_PATH, f"{model_name}_{PLACEMENT}{'_fine' if fine else ''}"
        )).eval().to(device)

        # There are two distance between microphones
        MAX_TAU = max_tau(R_RECEIVERS, SAMPLING_FREQUENCY)
        MAX_TAU_DIAG = max_tau(R_RECEIVERS, SAMPLING_FREQUENCY, diag=True)

        self.ls_model = models.LS(device, SAMPLING_FREQUENCY).eval()
        self.pso_model = models.PSO(device, SAMPLING_FREQUENCY).eval()


        self.cc_model = models.CC(
            SAMPLING_FREQUENCY,
            max_tau=MAX_TAU,
            tdoa_estimate=True
        ).eval()

        self.cc_diag_model = models.CC(
            SAMPLING_FREQUENCY,
            max_tau=MAX_TAU_DIAG,
            tdoa_estimate=True
        ).eval()


        self.gcc_model = models.GCC(
            device,
            max_tau=MAX_TAU,
            tdoa_estimate=True
        ).eval()

        self.gcc_diag_model = models.GCC(
            device,
            max_tau=MAX_TAU_DIAG,
            tdoa_estimate=True
        ).eval()


        self.tde_ild_model = models.TDE_ILD(
            device, MAX_TAU, SAMPLING_FREQUENCY, R_RECEIVERS).eval()
        
        self.srp_phat_model = models.SRP_PHAT(
            device, SAMPLING_FREQUENCY, MAX_TAU).eval()

        # Trained models
        self.pgcc_phat_model_0: models.PGCC_PHAT = load_model('pgcc-phat-0')
        self.pgcc_phat_model_1: models.PGCC_PHAT = load_model('pgcc-phat-1')
        self.pgcc_phat_model_2: models.PGCC_PHAT = load_model('pgcc-phat-2')

        self.ngcc_phat_model_0: models.NGCC_PHAT = load_model('ngcc-phat-0')
        self.ngcc_phat_model_1: models.NGCC_PHAT = load_model('ngcc-phat-1')
        self.ngcc_phat_model_2: models.NGCC_PHAT = load_model('ngcc-phat-2')

        self.e2e_cnn_model: models.E2E_CNN = load_model('e2e-cnn')

        self.mlat = self.pso_model if MLAT == 'PSO' else self.ls_model


    def init_data(self, data: DatasetInfo) -> None:
        receivers = data['receivers'].to(self.device)
        room_dimensions = data['room-dimensions']

        self.srp_phat_model.init(receivers, room_dimensions)
        self.ls_model.init(receivers, room_dimensions)
        self.pso_model.init(receivers, room_dimensions)


    @torch.no_grad()
    def estimate_tdoa(self,
        signal: torch.Tensor,
        f1: Callable[[torch.Tensor], torch.Tensor],
        f2: Callable[[torch.Tensor], torch.Tensor],
        f3: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        s1, s2, s3, shape = split_signal_tdoa(signal)
        return merge_signal_tdoa(
            f1(s1),
            f2(s2),
            f3(s3),
            shape, #type: ignore
        )

    """""""""""""""""""""""""""
    """""" TDOA methods """""""
    """""""""""""""""""""""""""
    def cc_tdoa(self, signal: torch.Tensor) -> torch.Tensor:
        return self.estimate_tdoa(
            signal,
            self.cc_model,
            self.cc_diag_model,
            self.cc_diag_model
        )


    def gcc_phat_tdoa(self, signal: torch.Tensor) -> torch.Tensor:
        return self.estimate_tdoa(
            signal,
            self.gcc_model,
            self.gcc_diag_model,
            self.gcc_diag_model
        )
  

    @torch.no_grad()
    def pgcc_phat_tdoa(self, signal: torch.Tensor) -> torch.Tensor:
        return self.estimate_tdoa(
            signal,
            lambda x: torch.round(self.pgcc_phat_model_0(x) \
                * self.pgcc_phat_model_0.normalization()),
            lambda x: torch.round(self.pgcc_phat_model_1(x) \
                * self.pgcc_phat_model_1.normalization()),
            lambda x: torch.round(self.pgcc_phat_model_2(x) \
                * self.pgcc_phat_model_2.normalization()),
        )
    
    
    @torch.no_grad()
    def ngcc_phat_tdoa(self, signal: torch.Tensor) -> torch.Tensor:
        return self.estimate_tdoa(
            signal,
            lambda x: torch.argmax(self.ngcc_phat_model_0(x), dim=-1) - self.ngcc_phat_model_0.get_max_tau(),
            lambda x: torch.argmax(self.ngcc_phat_model_1(x), dim=-1) - self.ngcc_phat_model_1.get_max_tau(),
            lambda x: torch.argmax(self.ngcc_phat_model_2(x), dim=-1) - self.ngcc_phat_model_2.get_max_tau(),
        )
    



    """""""""""""""""""""""""""
    """""" LOC methods """"""""
    """""""""""""""""""""""""""
    def ls_pso_mlat(self, tdoas: torch.Tensor) -> torch.Tensor:
        return self.mlat(tdoas)
    

    def cc(self, signal: torch.Tensor) -> torch.Tensor:
        return self.ls_pso_mlat(self.cc_tdoa(signal))


    def gcc_phat(self, signal: torch.Tensor) -> torch.Tensor:
        return self.ls_pso_mlat(self.gcc_phat_tdoa(signal))
    

    def pgcc_phat(self, signal: torch.Tensor) -> torch.Tensor:
        return self.ls_pso_mlat(self.pgcc_phat_tdoa(signal))


    def ngcc_phat(self, signal: torch.Tensor) -> torch.Tensor:
        return self.ls_pso_mlat(self.ngcc_phat_tdoa(signal))


    def tde_ild(self, signal: torch.Tensor) -> torch.Tensor:
        return self.tde_ild_model(signal)


    def e2e_cnn(self, signal: torch.Tensor) -> torch.Tensor:
        return self.e2e_cnn_model(signal)
  

    def srp_phat(self, signal: torch.Tensor) -> torch.Tensor:
        return self.srp_phat_model(signal)
