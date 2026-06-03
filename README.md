# KerogenAnalysis
!!!TODO!!!

export DATA_PATH="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"

python -m gex.extract_krg_trajectory_to_file ${DATA_PATH}/type1.ch4.300.gro ${DATA_PATH}/trj_krg/krg_99.gro --select KRG:99
python -m gex.corrfunc_krg_mol_plotter ${DATA_PATH}/trj_krg/krg_99.gro ${DATA_PATH}/figs/corrfunc_krg_99.svg

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
