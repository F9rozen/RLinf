#! /bin/bash
#
# Usage:
#   ./eval_embodiment.sh [CONFIG_NAME] [ROBOT_PLATFORM]
#
# GimArm + DreamZero (shortcut CONFIG_NAME):
#   ./eval_embodiment.sh gimarm
#   ./eval_embodiment.sh gimarm_eval_dreamzero GIMARM
# Override checkpoint / tokenizer:
#   ./eval_embodiment.sh gimarm_eval_dreamzero GIMARM \
#     actor.model.model_path=/path/to/dz22-gimarm actor.model.tokenizer_path=/path/to/umt5-xxl

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        gimarm|gim_arm)
            CONFIG_NAME="gimarm_eval_dreamzero"
            ;;
        *)
            CONFIG_NAME=$1
            ;;
    esac
fi

# NOTE: Robot platform (LIBERO, GIMARM, ALOHA, BRIDGE, …). Default follows CONFIG_NAME when unset.
if [ -z "$2" ]; then
    if [ "$CONFIG_NAME" == "gimarm_eval_dreamzero" ]; then
        ROBOT_PLATFORM=${ROBOT_PLATFORM:-GIMARM}
    else
        ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
    fi
else
    ROBOT_PLATFORM=$2
fi

export ROBOT_PLATFORM

ROBOT_PLATFORM_UPPER=$(echo "${ROBOT_PLATFORM}" | tr '[:lower:]' '[:upper:]')

# Libero variant: standard, pro, plus (only when evaluating Libero)
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$ROBOT_PLATFORM_UPPER" == "LIBERO" ]; then
    if [ "$LIBERO_TYPE" == "pro" ]; then
        export LIBERO_PERTURBATION="all"  # all,swap,object,lan
        echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
    elif [ "$LIBERO_TYPE" == "plus" ]; then
        export LIBERO_SUFFIX="all"
        echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
    else
        echo "Evaluation Mode: Standard LIBERO"
    fi
elif [ "$ROBOT_PLATFORM_UPPER" == "GIMARM" ] || [ "$ROBOT_PLATFORM_UPPER" == "GIM_ARM" ]; then
    echo "Evaluation Mode: GimArm (RealWorldEnv; see config/env/gimarm.yaml, override_cfg.is_dummy for hardware)"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"

# Positional $1=config (or alias), $2=ROBOT_PLATFORM; remaining args passed to Hydra.
EXTRA_ARGS=("${@:3}")
CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
)
CMD+=("${EXTRA_ARGS[@]}")
echo "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${MEGA_LOG_FILE}"
