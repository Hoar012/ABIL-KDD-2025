# Generate data
python data_generator.py n=1000 task=packing-shapes  mode=train
python cliport_src/data_generator.py n=1000 task=put-block-in-bowl-seen-colors  mode=test

# Train
jac-run trainval-cliport-bc.py cliport packing-shapes --iterations 1000

# Eval
jac-run eval-cliport.py n=100 task=packing-shapes model=bc  mode=test load=$ckp