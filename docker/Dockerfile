# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications for Sydney Uni:
# Set base image to be ubuntu 16.04 and cuda10.2 to run on our HPC
# contact sih.info@sydney.edu.au
# Conda version now hardcoded.
#
# Build with:
# sudo docker build -t sydneyinformaticshub/alphafold .
# python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
#
# Push with:
# sudo docker push sydneyinformaticshub/alphafold
#
# Run with:
# sudo docker run -it --gpus=all sydneyinformaticshub/alphafold /bin/bash -c "python /app/alphafold/alphafold-2.3.2/run_alphafold.py \
#   --data_dir=${ALPHADB} \
#   --uniref90_database_path=${ALPHADB}/uniref90/uniref90.fasta \
#   --mgnify_database_path=${ALPHADB}/mgnify/mgy_clusters_2018_12.fa \
#   --template_mmcif_dir=${ALPHADB}/pdb_mmcif/mmcif_files/ \
#   --obsolete_pdbs_path=${ALPHADB}/pdb_mmcif/obsolete.dat \
#   --fasta_paths=/project/Training/DATA/input.fasta \
#   --output_dir=${WORKDIR}/output_directory_gpu \
#   --db_preset=full_dbs \
#   --max_template_date=2022-03-23 \
#   --use_gpu_relax=True \
#   --model_preset=monomer \
#   --pdb70_database_path=${ALPHADB}/pdb70/pdb70 \
#   --bfd_database_path=${ALPHADB}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
#   --uniref30_database_path=${ALPHADB}/uniclust30/uniclust30_2018_08/uniclust30_2018_0 \
#  "
# export SINGULARITY_CACHEDIR=`pwd`
# export SINGULARITY_TMPDIR=`pwd`
# singularity build alpha.img docker://sydneyinformaticshub/alphafold

FROM nbutter/pytorch:ubuntu1604

RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py37_4.10.3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py37_4.10.3-Linux-x86_64.sh

# There is another version of conda in the base image, remove it.
RUN rm -rf /build/miniconda3
ENV PATH="/opt/conda/bin:$PATH"

RUN conda install -y pip

# Version 1.2.0
# https://github.com/YoshitakaMo/localcolabfold/tree/0ad333ffa24a9896361a54d8d2e18a493ec68184
ENV COLABFOLDDIR="/opt/colabfold"

WORKDIR ${COLABFOLDDIR}

RUN wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt --no-check-certificate

# Install the conda/pip packages
RUN conda install -c conda-forge cudatoolkit==10.2.89 -y
RUN conda install -c conda-forge openmm==7.5.0 -y
RUN conda install -c conda-forge pdbfixer==1.7 -y

# Apply OpenMM patch.
RUN wget -qnc https://raw.githubusercontent.com/deepmind/alphafold/main/docker/openmm.patch --no-check-certificate && \
 cd /opt/conda/lib/python3.7/site-packages && patch -s -p0 < ${COLABFOLDDIR}/openmm.patch && \
 rm ${COLABFOLDDIR}/openmm.patch

RUN conda install -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0 -y
RUN pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
RUN pip install -U https://storage.googleapis.com/jax-releases/cuda102/jaxlib-0.1.71+cuda102-cp37-none-manylinux2010_x86_64.whl
RUN pip install jax==0.2.25


RUN mkdir -p ${COLABFOLDDIR}/bin && \
  echo $'#!/bin/bash \n\
export TF_FORCE_UNIFIED_MEMORY="1" \n\
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0" \n\
export COLABFOLDDIR=$COLABFOLDDIR \n\
export XDG_CACHE_HOME="${COLABFOLDDIR}" \n\
export PATH="/opt/conda/bin:$PATH" \n\
/opt/conda/bin/colabfold_batch $@' > ${COLABFOLDDIR}/bin/colabfold

#ENV /opt/conda/lib/python3.7/site-packages/colabfold
ENV PATH="/opt/colabfold/bin:$PATH"

# hack to share the parameter files in a workstation.
RUN cd /opt/conda/lib/python3.7/site-packages/colabfold && \
sed -i -e "s#props_path = \"stereo_chemical_props.txt\"#props_path = \"${COLABFOLDDIR}/stereo_chemical_props.txt\"#" batch.py
# Use gpu-accelerated relaxation.
RUN cd /opt/conda/lib/python3.7/site-packages/alphafold/relax && \
sed -i -e 's/CPU/CUDA/g' amber_minimize.py
# Use 'Agg' for non-GUI backend
RUN cd /opt/conda/lib/python3.7/site-packages/colabfold && \
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py
