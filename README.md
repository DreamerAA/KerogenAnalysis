# KerogenAnalysis
!!!TODO!!!

1) у меня есть скрипты extract_gas_tajectory_to_file, extract_krg_trajectory_to_file они делаю схожую вещь, Берут файл парсят его и сохраняют в другой файл часть информации.
файл extract_gas_trajectory_to_file болле старый, думаю можно вынести общие вещи в utils или base. и сделать скрипт работающим не по циклу а по входным путям как в extract_krg_trajectory_to_file
2) Второй шаг это извлечени определенных структур их четнеи и сохранение. Файл dynamic_struct_extractor.py сейчас делает больше чем нужно, он не просто извлекает структуру, но и бинаризует ее, хочу разделить на несколько скриптов эту логику. Для начала просто вытащим структуру, и в отдельном скрипт делаем логику бинаризации (binarization_structs - что-нибудь такое), и посмотри также на скрипт dynamic_float_struct_extractor - он делает схожую вещь, но сохраняет не бинаризацию и дистанс мап - наверное нужен ренейм файла. 

3) обрати внимание на изменяемых тобою файлах сделать логику подачи путей через входные аргументы скрипта, желательно без дефлотных путей, чтобы в будущем, когда это будет наружу торчать не было абсолютных путей в скриптах


## Check main logic

export DATA_PATH="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"

## Экстракция траектории молекулы газа
python -m gex.extract_gas_trajectory_to_file $DATA_PATH/type1.ch4.300.gro $DATA_PATH/trj.gro


## Экстракция фреймов структур
python -m gex.dynamic_struct_extractor $DATA_PATH/type1.ch4.300.gro $DATA_PATH/structures

python3 -m gex.dynamic_struct_extractor $DATA_PATH/type1.ch4.300.gro $DATA_PATH/structures --auto-indexes --mode all --count-structures 5 --dry-run
python3 -m gex.dynamic_struct_extractor $DATA_PATH/type1.ch4.300.gro $DATA_PATH/structures --auto-indexes --mode all --count-structures 500

## Бинаризация
python3 -m gex.binarization_structs $DATA_PATH/structures $DATA_PATH/bin_images $DATA_PATH/raw_images --ref-size 250 --mode all --count-slices 500 --num_workers 10

## Экстракция молекулы керогена

python -m gex.extract_krg_trajectory_to_file ${DATA_PATH}/type1.ch4.300.gro ${DATA_PATH}/trj_krg/krg_99.gro --select KRG:99


## Построение Корреляционной функции молекулы керогена

python -m gex.corrfunc_krg_mol_plotter ${DATA_PATH}/trj_krg/krg_99.gro ${DATA_PATH}/figs/corrfunc_krg_99.svg




## Экстракция DM 
python3 -m gex.distance_map_structs $DATA_PATH/type1.ch4.300.gro $DATA_PATH/structures --auto-indexes --mode all --count-structures 500




## Визуализация молекулы керогена

python -m gex.vis_slice_struct "$DATA_PATH/images/result-img-num=25000_time-ps=50_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000.npy" "$DATA_PATH/figs/img_slice_num=25000.svg" --ref_size 0.2 --size 1

python -m gex.vis_slice_struct "$DATA_PATH/images/result-img-num=551025000_time-ps=1102050_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000.npy" "$DATA_PATH/figs/img_slice_num=551025000.svg" --ref_size 0.2 --size 1

python -m gex.vis_slice_struct "$DATA_PATH/images/result-img-num=1102025000_time-ps=2204050_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000.npy" "$DATA_PATH/figs/img_slice_num=1102025000.svg" --ref_size 0.2 --size 1



export PYMOL_SCRIPT_ARGS="--ker-pdb '$DATA_PATH/ker.pdb' --sim-gro '$DATA_PATH/type1.ch4.300.gro' --frame 5 --mol-index 85 --box-size 30"
pymol -q external_scripts/visualize_kerogen_part_cell_with_molecula.pml
`turn y, 45 & turn x, 15`


export PYMOL_SCRIPT_ARGS="--ker-pdb '$DATA_PATH/ker.pdb' --sim-gro '$DATA_PATH/type1.ch4.300.gro' --frame 15 --mol-index 85 --box-size 30"
pymol -q external_scripts/visualize_kerogen_part_cell_with_molecula.pml
`turn y, 45 & turn x, 15`

export PYMOL_SCRIPT_ARGS="--ker-pdb '$DATA_PATH/ker.pdb' --sim-gro '$DATA_PATH/type1.ch4.300.gro' --frame 50 --mol-index 85 --box-size 30"
pymol -q external_scripts/visualize_kerogen_part_cell_with_molecula.pml
`turn y, 45 & turn x, 15`
