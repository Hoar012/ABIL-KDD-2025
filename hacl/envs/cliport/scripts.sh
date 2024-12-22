export CLIPORT_ROOT=$(pwd)

# generate data
python data_generator.py n=100 task=packing-shapes  mode=test

jac-run eval-cliport-bc.py n=4 task=packing-shapes  mode=test record.save_video=True