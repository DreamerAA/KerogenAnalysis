# KerogenAnalysis
!!!TODO!!!


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

python -m gex.corrfunc_krg_mol_plotter ${DATA_PATH}/trj_krg/krg_99.gro ${DATA_PATH}/figs/corrfunc_krg_99.svg ${DATA_PATH}/msd/krg_99.pickle

## Построение Корреляционной функции структуры керогена

python -m gex.corrfunc_struct_plotter ${DATA_PATH}/bin_images ${DATA_PATH}/ct_pore.npy ${DATA_PATH}/figs/corrfunc.svg --trj ${DATA_PATH}/../ch4/trj.gro:CH4 --trj ${DATA_PATH}/../h2/trj.gro:H2 --pore --max-t 2.8

## PNM экстракция

export EXTRACTOR_PATH="/media/andrey/Samsung_T5/DCore/SSM-2/pore-network-extraction_new/build/clang-15-release-cpu/bin/extractor_example"
export EXTRACTOR_CONFIG="/media/andrey/Samsung_T5/DCore/SSM-2/pore-network-extraction_new/build/clang-15-release-cpu/example/config/ExtractorExampleConfig.json"

python -m gex.pnm_extractor $DATA_PATH $EXTRACTOR_PATH $EXTRACTOR_CONFIG


## Распределения по PNM

python -m gex.distr_pnm_element_size_plotter $DATA_PATH/pnm $DATA_PATH/figs --label 'CH4' --pnm-step 120 --x-max 1.7



## PIL распределения (одна директория)

python -m gex.generate_pil_distr $DATA_PATH/pnm $DATA_PATH

python -m gex.pil_plotter $DATA_PATH






# OLD BUT NEED CHECK


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

python -m gex.distr_pnm_element_size_plotter $DATA_PATH/pnm $DATA_PATH/figs --label 'CH4'


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
