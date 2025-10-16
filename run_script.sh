# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_dataset.py -v -d herring --hvgs --per_cell_type -s remove_recovery &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_dataset.py -v -d herring_gaba --hvgs --per_cell_type -s three_interpolation &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_dataset.py -v -d wot -s remove_recovery &
nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_continuous_tps.py -v -d herring --hvgs --per_cell_type -s remove_recovery --normalize &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_continuous_tps.py -v -d herring_gaba --hvgs --per_cell_type -s three_interpolation --normalize &
# nohup bash ../discord.sh PYTHONPATH=".:$PYTHONPATH" python benchmark/test_dataset.py -v -d herring_gaba --hvgs --per_cell_type -s three_interpolation &
