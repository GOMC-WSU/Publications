{% extends "slurm.sh" %}

{% block header %}
{% set gpus = operations|map(attribute='directives.ngpu')|sum %}
    {{- super () -}}

{% if gpus %}
#SBATCH -q gpu
#SBATCH --gres gpu:{{ gpus }}
#SBATCH --constraint=v100
{%- else %}
#SBATCH -q primary
#SBATCH --constraint=intel
{%- endif %}

#SBATCH -N 1
#SBATCH --mail-type=ALL

echo  "Running on host" hostname
echo  "Time is" date

conda activate 2022_crawford_1

module load python/3.9
module swap gnu7 intel/2019

{% if gpus %}
module load cuda/11.0
{%- endif %}

{% endblock header %}

{% block body %}
    {{- super () -}}


{% endblock body %}
