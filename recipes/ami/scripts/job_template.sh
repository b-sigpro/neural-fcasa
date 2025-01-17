#!/bin/bash

#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load singularitypro
module load hpcx/2.12

export SINGULARITY_BINDPATH="$SINGULARITY_BINDPATH,/groups,/scratch"

export SINGULARITYENV_COLUMNS=120
export SINGULARITYENV_PYTHONUNBUFFERED=true

export SINGULARITYENV_OPAL_PREFIX=/usr/local/ 
export SINGULARITYENV_PMIX_INSTALL_PREFIX=/usr/local/
