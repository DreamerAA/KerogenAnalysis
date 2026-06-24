# KerogenAnalysis
!!!TODO!!!

## Использование скриптов 
+ atom_visualization.py
+ binarization_structs.py
complexity_estimation.py
+ corrfunc_krg_mol_plotter.py
+ corrfunc_struct_plotter.py
distance_map_structs.py
distr_pnm_connectivity.py
distr_pnm_count_pores_throats.py
distr_pnm_element_size_plotter.py
+ dynamic_struct_extractor.py
errors_params.py
euler_distribution.py
+ extract_gas_trajectory_to_file.py
+ extract_krg_trajectory_to_file.py
extract_one_molecula.py
find_best_params.py
+ generate_pil_distr.py
msdt_builder.py
+ pil_plotter.py
+ pnm_extractor.py
powerlaw_analysis.py
sim_algo_check.py
simulate_trajectory.py
stationarity.py
structure_image_utils.py
+ trap_distr_builder.py
vis_atoms_struct.py
vis_slice_struct.py
vis_struct_pnm.py
vis_struct_trajectory.py
vis_traject.py


## Базовые скрипты расчетов и визуализация 

export DATA_PATH="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"

### Экстракция траектории молекулы газа
python -m gex.extract_gas_trajectory_to_file $DATA_PATH/type1.ch4.300.gro $DATA_PATH/trj.gro


### Экстракция фреймов структур
python -m gex.dynamic_struct_extractor $DATA_PATH/type1.ch4.300.gro $DATA_PATH/structures

python3 -m gex.dynamic_struct_extractor $DATA_PATH/type1.ch4.300.gro $DATA_PATH/structures --auto-indexes --mode all --count-structures 5 --dry-run
python3 -m gex.dynamic_struct_extractor $DATA_PATH/type1.ch4.300.gro $DATA_PATH/structures --auto-indexes --mode all --count-structures 500

### Бинаризация
python3 -m gex.binarization_structs $DATA_PATH/structures $DATA_PATH/bin_images $DATA_PATH/raw_images --ref-size 250 --mode all --count-slices 500 --num_workers 10

### Экстракция DM 

#### All
python3 -m gex.distance_map_structs $DATA_PATH_CH4/structures $DATA_PATH_CH4/float_images --mode all --count-slices 500 --ref-size 250

#### Part
python3 -m gex.distance_map_structs $DATA_PATH_CH4/structures $DATA_PATH_CH4/float_images --mode part --count-slices 10 --ref-size 250


### Экстракция молекулы керогена

python -m gex.extract_krg_trajectory_to_file ${DATA_PATH}/type1.ch4.300.gro ${DATA_PATH}/trj_krg/krg_99.gro --select KRG:99

### Построение Корреляционной функции молекулы керогена

python -m gex.corrfunc_krg_mol_plotter ${DATA_PATH}/trj_krg/krg_99.gro ${DATA_PATH}/figs/corrfunc_krg_99.svg ${DATA_PATH}/msd/krg_99.pickle

### Построение Корреляционной функции структуры керогена

python -m gex.corrfunc_struct_plotter ${DATA_PATH}/bin_images ${DATA_PATH}/ct_pore.npy ${DATA_PATH}/../figs/corrfunc.svg --trj ${DATA_PATH}/../ch4/trj.gro:CH4 --trj ${DATA_PATH}/../h2/trj.gro:H2 --pore --max-t 2.8

### PNM экстракция

export EXTRACTOR_PATH="/media/andrey/Samsung_T5/DCore/SSM-2/pore-network-extraction_new/build/clang-15-release-cpu/bin/extractor_example"
export EXTRACTOR_CONFIG="/media/andrey/Samsung_T5/DCore/SSM-2/pore-network-extraction_new/build/clang-15-release-cpu/example/config/ExtractorExampleConfig.json"

python -m gex.pnm_extractor $DATA_PATH $EXTRACTOR_PATH $EXTRACTOR_CONFIG


### Распределения по PNM

python -m gex.distr_pnm_element_size_plotter $DATA_PATH/pnm $DATA_PATH/figs --label 'CH4' --pnm-step 120 --x-max 1.7


### PIL распределения (одна директория)

python -m gex.generate_pil_distr $DATA_PATH/pnm $DATA_PATH --x-min 0.025
python -m gex.pil_plotter $DATA_PATH --x-min 0.025


## Распределение времён ловушек
export DATA_PATH_CH4="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"
export DATA_PATH_H2="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/"
python -m gex.trap_distr_builder $DATA_PATH_CH4/ --label CH4 --num 1 --output $DATA_PATH_CH4/figs/Pt_loglog.svg
python -m gex.trap_distr_builder $DATA_PATH_H2/ --label H2 --num 2 --output $DATA_PATH_H2/figs/Pt_loglog.svg


## Power law analysis

export DATA_PATH_CH4="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"
export DATA_PATH_H2="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/h2/"
python -m gex.powerlaw_analysis $DATA_PATH_CH4 --prefix SIB --mode xmin --n_synth 2500
python -m gex.powerlaw_analysis $DATA_PATH_H2 --prefix SIB --mode xmin --n_synth 2500


## Дополнительная визуализация для анализа

### Визуализация атомов

python -m gex.atom_visualization

### Визуализация траектория и структура
export DATA_PATH_CH4="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"
export IMG1="$DATA_PATH_CH4/float_images/result-img-num=25000_time-ps=50_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000.npy"
export IMG2="$DATA_PATH_CH4/float_images/result-img-num=3275000_time-ps=6550_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000.npy"


python -m gex.vis_struct_trajectory $DATA_PATH_CH4 "$IMG1" --num 2
python -m gex.vis_struct_trajectory $DATA_PATH_CH4 "$IMG2" --num 2









# OLD BUT NEED CHECK





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


# NEW NEED CHECK


## Распределения по PNM

python -m gex.distr_pnm_connectivity \
  --pnm ${DATA_PATH}/../ch4/pnm:type1-300K-CH4 \
  --pnm ${DATA_PATH}/../h2/pnm:type1-300K-H2

python -m gex.distr_pnm_count_pores_throats \
  --pnm $DATA_PATH/pnm:type1-300K-CH4 \
  --pnm ${DATA_PATH/ch4/h2}/pnm:type1-300K-H2


## Извлечение молекулы керогена из PDB

python -m gex.extract_one_molecula $DATA_PATH/Ker.pdb $DATA_PATH/KRG_chainA_res1.pdb --chain A --resid 1


## Числа Эйлера (PNM vs image)

python -m gex.euler_distribution $DATA_PATH --config $EXTRACTOR_CONFIG --extractor $EXTRACTOR_PATH





## Распределения по PNM

python -m gex.distr_pnm_connectivity \
  --trj $DATA_PATH/pnm:type1-300K-CH4 \
  --trj ${DATA_PATH/ch4/h2}/pnm:type1-300K-H2

python -m gex.distr_pnm_count_pores_throats \
  --trj $DATA_PATH/pnm:type1-300K-CH4 \
  --trj ${DATA_PATH/ch4/h2}/pnm:type1-300K-H2



## MSD (mean square displacement)

python -m gex.msdt_builder \
  --trj $DATA_PATH/trj.gro:type1-300K-CH4:1 \
  --trj ${DATA_PATH/ch4/h2}/trj.gro:type1-300K-H2:2


## Стационарность PNM

python -m gex.stationarity $DATA_PATH


## Распределение времён ловушек

python -m gex.trap_distr_builder $DATA_PATH --label CH4 --num 1
python -m gex.trap_distr_builder ${DATA_PATH/ch4/h2} --label H2 --num 2


## Оценка сложности алгоритмов

python -m gex.complexity_estimation $DATA_PATH $DATA_PATH/complexity.pdf


## Визуализация траектории (3D)

python -m gex.vis_traject $DATA_PATH/trj.gro 2
python -m gex.vis_traject $DATA_PATH/trj.gro 2 --traps $DATA_PATH/traps/SIB/traps_2.pickle

python -m gex.simulate_trajectory $DATA_PATH


## Проверка алгоритмов

python -m gex.sim_algo_check $DATA_PATH
python -m gex.errors_params $DATA_PATH
python -m gex.find_best_params $DATA_PATH/errors/find_best_params


## Визуализация структуры + изображение

export IMG="$DATA_PATH/float_images/result-img-num=551025000_time-ps=1102050_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000.npy"

python -m gex.vis_struct_trajectory $DATA_PATH "$IMG" --num 2
python -m gex.vis_atoms_struct $DATA_PATH "$IMG" --index 551025000 --time-ps 1102050
python -m gex.vis_struct_pnm "$IMG" $DATA_PATH/pnm/pnm-num=551025000_time-ps=1102050_bbox=...
