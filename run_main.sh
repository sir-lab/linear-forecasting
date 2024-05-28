# No instance norm:
# ETTm1
python main.py --dataset ETTm1 --context_length 720 --horizon 96 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTm1 --context_length 720 --horizon 192 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTm1 --context_length 720 --horizon 336 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTm1 --context_length 720 --horizon 720 --alpha 0.00001 --no-instance_norm --no-individual
# ETTm2
python main.py --dataset ETTm2 --context_length 720 --horizon 96 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTm2 --context_length 720 --horizon 192 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTm2 --context_length 720 --horizon 336 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTm2 --context_length 720 --horizon 720 --alpha 0.00001 --no-instance_norm --no-individual
# ETTh1
python main.py --dataset ETTh1 --context_length 720 --horizon 96 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTh1 --context_length 720 --horizon 192 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTh1 --context_length 720 --horizon 336 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTh1 --context_length 720 --horizon 720 --alpha 0.00001 --no-instance_norm --no-individual
# ETTh2
python main.py --dataset ETTh2 --context_length 720 --horizon 96 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTh2 --context_length 720 --horizon 192 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTh2 --context_length 720 --horizon 336 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset ETTh2 --context_length 720 --horizon 720 --alpha 0.00001 --no-instance_norm --no-individual
# Exchange
python main.py --dataset exchange --context_length 720 --horizon 96 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset exchange --context_length 720 --horizon 192 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset exchange --context_length 720 --horizon 336 --alpha 0.00001 --no-instance_norm --no-individual
python main.py --dataset exchange --context_length 720 --horizon 720 --alpha 0.00001 --no-instance_norm --no-individual
# Weather
python main.py --dataset weather --context_length 720 --horizon 96 --alpha 0.00001 --no-instance_norm --individual   # Same setting as DLinear and FITS
python main.py --dataset weather --context_length 720 --horizon 192 --alpha 0.00001 --no-instance_norm --individual   # Same setting as DLinear and FITS
python main.py --dataset weather --context_length 720 --horizon 336 --alpha 0.00001 --no-instance_norm --individual   # Same setting as DLinear and FITS
python main.py --dataset weather --context_length 720 --horizon 720 --alpha 0.00001 --no-instance_norm --individual   # Same setting as DLinear and FITS
# ECL
python main.py --dataset electricity --context_length 720 --horizon 96 --alpha 0.00001 --no-instance_norm --no-individual --max_train_N 1000000
python main.py --dataset electricity --context_length 720 --horizon 192 --alpha 0.00001 --no-instance_norm --no-individual --max_train_N 1000000
python main.py --dataset electricity --context_length 720 --horizon 336 --alpha 0.00001 --no-instance_norm --no-individual --max_train_N 1000000
python main.py --dataset electricity --context_length 720 --horizon 720 --alpha 0.00001 --no-instance_norm --no-individual --max_train_N 1000000
# Traffic
python main.py --dataset traffic --context_length 720 --horizon 96 --alpha 0.00001 --no-instance_norm --no-individual --max_train_N 1000000
python main.py --dataset traffic --context_length 720 --horizon 192 --alpha 0.00001 --no-instance_norm --no-individual --max_train_N 1000000
python main.py --dataset traffic --context_length 720 --horizon 336 --alpha 0.00001 --no-instance_norm --no-individual --max_train_N 1000000
python main.py --dataset traffic --context_length 720 --horizon 720 --alpha 0.00001 --no-instance_norm --no-individual --max_train_N 1000000
####################################################################################################################################
# With instance norm:
# ETTm1
python main.py --dataset ETTm1 --context_length 720 --horizon 96 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTm1 --context_length 720 --horizon 192 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTm1 --context_length 720 --horizon 336 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTm1 --context_length 720 --horizon 720 --alpha 0.00001 --instance_norm --no-individual
# ETTm2
python main.py --dataset ETTm2 --context_length 720 --horizon 96 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTm2 --context_length 720 --horizon 192 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTm2 --context_length 720 --horizon 336 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTm2 --context_length 720 --horizon 720 --alpha 0.00001 --instance_norm --no-individual
# ETTh1
python main.py --dataset ETTh1 --context_length 720 --horizon 96 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTh1 --context_length 720 --horizon 192 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTh1 --context_length 720 --horizon 336 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTh1 --context_length 720 --horizon 720 --alpha 0.00001 --instance_norm --no-individual
# ETTh2
python main.py --dataset ETTh2 --context_length 720 --horizon 96 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTh2 --context_length 720 --horizon 192 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTh2 --context_length 720 --horizon 336 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset ETTh2 --context_length 720 --horizon 720 --alpha 0.00001 --instance_norm --no-individual
# Exchange
python main.py --dataset exchange --context_length 720 --horizon 96 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset exchange --context_length 720 --horizon 192 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset exchange --context_length 720 --horizon 336 --alpha 0.00001 --instance_norm --no-individual
python main.py --dataset exchange --context_length 720 --horizon 720 --alpha 0.00001 --instance_norm --no-individual
# Weather
python main.py --dataset weather --context_length 720 --horizon 96 --alpha 0.00001 --instance_norm --individual  # Same setting as DLinear and FITS
python main.py --dataset weather --context_length 720 --horizon 192 --alpha 0.00001 --instance_norm --individual  # Same setting as DLinear and FITS
python main.py --dataset weather --context_length 720 --horizon 336 --alpha 0.00001 --instance_norm --individual  # Same setting as DLinear and FITS
python main.py --dataset weather --context_length 720 --horizon 720 --alpha 0.00001 --instance_norm --individual  # Same setting as DLinear and FITS
# ECL
python main.py --dataset electricity --context_length 720 --horizon 96 --alpha 0.00001 --instance_norm --no-individual --max_train_N 1000000
python main.py --dataset electricity --context_length 720 --horizon 192 --alpha 0.00001 --instance_norm --no-individual --max_train_N 1000000
python main.py --dataset electricity --context_length 720 --horizon 336 --alpha 0.00001 --instance_norm --no-individual --max_train_N 1000000
python main.py --dataset electricity --context_length 720 --horizon 720 --alpha 0.00001 --instance_norm --no-individual --max_train_N 1000000
# Traffic
python main.py --dataset traffic --context_length 720 --horizon 96 --alpha 0.00001 --instance_norm --no-individual --max_train_N 1000000
python main.py --dataset traffic --context_length 720 --horizon 192 --alpha 0.00001 --instance_norm --no-individual --max_train_N 1000000
python main.py --dataset traffic --context_length 720 --horizon 336 --alpha 0.00001 --instance_norm --no-individual --max_train_N 1000000
python main.py --dataset traffic --context_length 720 --horizon 720 --alpha 0.00001 --instance_norm --no-individual --max_train_N 1000000


