python nectar/sim-emotion-fedavg.py --num_cpus 2 --num_gpus 2 --desc "fedavg emotion baseline test" --critical_value 0.05 --mi_type "mi_cat" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-fedistill.py --num_cpus 2 --num_gpus 2 --desc "feddistill emotion baseline test" --critical_value 0.05 --mi_type "mi_cat" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion cat 0.05 distill" --critical_value 0.05 --mi_type "mi_cat" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion cat 0.1 distill" --critical_value 0.1 --mi_type "mi_cat" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion cat 0.15 distill" --critical_value 0.15 --mi_type "mi_cat" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion cat 0.2 distill" --critical_value 0.2 --mi_type "mi_cat" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion cat 0.25 distill" --critical_value 0.25 --mi_type "mi_cat" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion gauss 0.05 distill" --critical_value 0.05 --mi_type "mi_gauss" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion gauss 0.1 distill" --critical_value 0.1 --mi_type "mi_gauss" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion gauss 0.15 distill" --critical_value 0.15 --mi_type "mi_gauss" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion gauss 0.2 distill" --critical_value 0.2 --mi_type "mi_gauss" --num_clients 10 --num_rounds 25
python nectar/sim-emotion-mifl.py --num_cpus 2 --num_gpus 2 --desc "mifl emotion gauss 0.25 distill" --critical_value 0.25 --mi_type "mi_gauss" --num_clients 10 --num_rounds 25
