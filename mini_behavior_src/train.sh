# Grounding
# PDSketch
jac-run mini_behavior_src/trainval-mini-mp.py mini_behavior install-a-printer --use-offline=yes --structure-mode full --action-loss-weight 1 --evaluate-interval 0 --iteration 500 --append-expr

# ABL
jac-run mini_behavior_src/trainval-mini-abl.py mini_behavior install-a-printer --use-offline=yes --structure-mode abl --action-loss-weight 0 --evaluate-interval 0 --iteration 500 --append-expr

# Imitation
# BC
jac-crun 1 mini_behavior_src/trainval-mini-bc.py mini_behavior install-a-printer --seed 33 --use-offline=yes --iteration 1000 --append-expr

# ABIL-BC
jac-crun 1 mini_behavior_src/trainval-inst-abil-bc.py mini_behavior install-a-printer --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 1000 --append-expr --load_domain g-mini-behavior/abl-install-a-printer-load=install.pth

jac-crun 1 mini_behavior_src/trainval-put-abil-bc.py mini_behavior CleaningShoes --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-CleaningShoes-load=CleaningShoes.pth
jac-crun 1 mini_behavior_src/trainval-put-abil-bc.py mini_behavior CollectMisplacedItems --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-CollectMisplacedItems-load=CollectMisplacedItems.pth
jac-crun 1 mini_behavior_src/trainval-put-abil-bc.py mini_behavior LayingWoodFloors --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-LayingWoodFloors-load=scratch.pth
jac-crun 1 mini_behavior_src/trainval-put-abil-bc.py mini_behavior MakingTea --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-MakingTea-load=MakingTea.pth
jac-crun 1 mini_behavior_src/trainval-put-abil-bc.py mini_behavior PuttingAwayDishesAfterCleaning --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-PuttingAwayDishesAfterCleaning-load=PuttingAwayDishesAfterCleaning.pth
jac-crun 1 mini_behavior_src/trainval-put-abil-bc.py mini_behavior SortingBooks --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-SortingBooks-load=scratch.pth
jac-crun 1 mini_behavior_src/trainval-put-abil-bc.py mini_behavior Throwing_away_leftovers --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-Throwing_away_leftovers-load=Throwing_away_leftovers.pth
jac-crun 1 mini_behavior_src/trainval-put-abil-bc.py mini_behavior WateringHouseplants --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-WateringHouseplants-load=WateringHouseplants.pth

jac-crun 1 mini_behavior_src/trainval-clean-abil-bc.py mini_behavior CleaningACar --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-CleaningACar-load=scratch.pth

# DT
jac-crun 1 mini_behavior_src/trainval-mini-dt.py mini_behavior install-a-printer --seed 33 --use-offline=yes --iteration 1000 --append-expr

# ABIL-DT
jac-crun 1 mini_behavior_src/trainval-inst-abil-dt.py mini_behavior install-a-printer --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 1000 --append-expr --load_domain g-mini-behavior/abl-install-a-printer-load=install.pth

jac-crun 1 mini_behavior_src/trainval-put-abil-dt.py mini_behavior CleaningShoes --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-CleaningShoes-load=CleaningShoes.pth

jac-crun 1 mini_behavior_src/trainval-clean-abil-dt.py mini_behavior CleaningACar --seed 33 --use-offline=yes --evaluate-interval 0 --iteration 30000 --append-expr --load_domain g-mini-behavior/abl-CleaningACar-load=scratch.pth