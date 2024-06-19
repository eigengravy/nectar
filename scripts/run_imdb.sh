python nectar/sim-fedavg-imdb.py --num_cpus 2 --num_gpus 2 --desc "fedavg imdb baseline test" --critical_value 0.05 --mi_type "mi_cat" --num_clients 10 --num_rounds 20
python nectar/sim-fedistill-imdb.py --num_cpus 2 --num_gpus 2 --desc "feddistill imdb baseline test" --critical_value 0.05 --mi_type "mi_cat" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb cat 0.05 distill" --critical_value 0.05 --mi_type "mi_cat" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb cat 0.1 distill" --critical_value 0.1 --mi_type "mi_cat" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb cat 0.15 distill" --critical_value 0.15 --mi_type "mi_cat" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb cat 0.2 distill" --critical_value 0.2 --mi_type "mi_cat" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb cat 0.25 distill" --critical_value 0.25 --mi_type "mi_cat" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb gauss 0.05 distill" --critical_value 0.05 --mi_type "mi_gauss" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb gauss 0.1 distill" --critical_value 0.1 --mi_type "mi_gauss" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb gauss 0.15 distill" --critical_value 0.15 --mi_type "mi_gauss" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb gauss 0.2 distill" --critical_value 0.2 --mi_type "mi_gauss" --num_clients 10 --num_rounds 20
python nectar/sim-mifl-imdb.py --num_cpus 2 --num_gpus 2 --desc "mifl imdb gauss 0.25 distill" --critical_value 0.25 --mi_type "mi_gauss" --num_clients 10 --num_rounds 20
