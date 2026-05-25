from collections.abc import Iterable
import custom_function as cuts_f
from pocket_coffea.lib.cut_definition import Cut

dileptonic_presel = Cut(
    name="dileptonic",
    params={
        "njet": 2,
        "nbjet": 0,
        "pt_leading_lepton": 25,
        "pt_subleading_lepton": 15,
        "met": 40,
        "m_ee_mumu_min": 20., 
        "dy_window":{
            "start": 76.,
            "stop": 106.
            }
    },
    function=cuts_f.dileptonic,
)

def custom_get_dilepton(selec):
    return Cut(
        name="dilepton_cut",
        params={"selec": selec},
        function=cuts_f.custom_dilepton,
    )
    



def get_ttB_id(ttBid):
    return Cut(
        name="ttB_cut",
        params={"ttBId": ttBid},
        function=cuts_f.ttB_masks,
    )

