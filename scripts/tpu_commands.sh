#! /bin/bash

function _tpu_ips {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    gcloud alpha compute tpus tpu-vm describe $tpu_name --zone $tpu_zone --project $tpu_project | grep -oP 'externalIp: \K(.+)$'

}

function _tpu_create {
    tpu_zone=$1
    tpu_project=$2
    tpu_gen=$3
    tpu_cores=$4
    tpu_name=$5
    if [ "$tpu_gen" = "v3" ]; then
        software_version='tpu-vm-base'
    else
        software_version='tpu-vm-v4-base'
    fi

    if [[ $tpu_cores =~ ^[0-9]+$ ]]; then
        gcloud alpha compute tpus tpu-vm create \
            $tpu_name \
            --accelerator-type="$tpu_gen-$tpu_cores" \
            --version $software_version \
            --zone $tpu_zone \
            --project $tpu_project
    else
        gcloud alpha compute tpus tpu-vm create \
            $tpu_name \
            --type="$tpu_gen" \
            --topology="$tpu_cores" \
            --version $software_version \
            --zone $tpu_zone \
            --project $tpu_project
    fi
}

function _tpu_retry_create {
    while true; do
        _tpu_create "$@"
        sleep 120s
    done
}

function _tpu_cp_ssh_key {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    gcloud alpha compute tpus tpu-vm scp \
        $HOME/.ssh/authorized_keys \
        $tpu_name:/home/$USER/.ssh/ \
        --worker=all \
        --project $tpu_project \
        --zone $tpu_zone
}

function _tpu_setup {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        scp $PROJECT_HOME/$PROJECT_NAME/scripts/tpu_vm_setup.sh $host:~/
        ssh $host '~/tpu_vm_setup.sh' &
    done
    wait &> /dev/null

    for host in $tpu_ips; do
        scp $PROJECT_HOME/$PROJECT_NAME/scripts/tpu_vm_setup.sh $host:~/
        ssh $host '~/tpu_vm_setup.sh' &
    done
    wait &> /dev/null
}

function _tpu_check {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        echo "============== Checking host: $host =============="
        ssh $host 'tmux capture-pane -pt launch -S -2000'
        echo
        echo
    done
}

function _tpu_copy {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        rsync -avPI --exclude=logs --exclude=__pycache__ --exclude=.git $PROJECT_HOME/$PROJECT_NAME $host:~/ &
    done
    wait &> /dev/null
    sleep 1s

    for host in $tpu_ips; do
        rsync -avPI --exclude=logs --exclude=__pycache__ --exclude=.git $PROJECT_HOME/$PROJECT_NAME $host:~/ &
    done
    wait &> /dev/null
    sleep 1s
}

function _tpu_stop {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh $host 'tmux kill-session -t launch ; pkill -9 python' &
    done
    wait &> /dev/null
}

function _tpu_launch {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    command=$4

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh $host "tmux new -d -s launch ~/$PROJECT_NAME/launcher/$command" &
    done
    wait &> /dev/null
}

function _tpu_maintain {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    gcloud alpha compute tpus tpu-vm simulate-maintenance-event $tpu_name \
        --project $tpu_project \
        --zone=$tpu_zone \
        --workers=all
}

function _tpu_ssh {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    command="$4"

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh $host "$command" &
    done
    wait &> /dev/null
}

function _tpu_reboot {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=$(_tpu_ips $tpu_zone $tpu_project $tpu_name)
    for host in $tpu_ips; do
        ssh $host 'sudo reboot' &
    done
    wait &> /dev/null
}


function tpu {
    trap "trap - SIGINT SIGTERM; return 1;" SIGINT SIGTERM


    # =============== TPU Project Specific Definitions ===============
    export PROJECT_HOME='<your project home directory (parent of EasyLM)'
    export PROJECT_NAME='EasyLM'
    tpu_zone='<tpu zone>'
    if [ "$1" = "<short name for your tpu project, you can define multiple ones>" ]; then
        tpu_project='<full name for your tpu project>'
        tpu_zone='us-east1-d'
        tpu_gen='v3'
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi
    # =============== End of TPU Project Specific Definitions ===============


    if [ "$2" = "list" ]; then
        gcloud alpha compute tpus tpu-vm list --zone $tpu_zone --project $tpu_project
    elif [ "$2" = "describe" ]; then
        gcloud alpha compute tpus tpu-vm describe $3 --zone $tpu_zone --project $tpu_project
    elif [ "$2" = "ips" ]; then
        _tpu_ips $tpu_zone $tpu_project $3
    elif [ "$2" = "delete" ]; then
        gcloud alpha compute tpus tpu-vm delete $3 --zone $tpu_zone --project $tpu_project --quiet
    elif [ "$2" = "delete_queued" ]; then
            gcloud alpha compute tpus queued-resources delete $3 --project $tpu_project --zone $tpu_zone
    elif [ "$2" = "create" ]; then
        _tpu_create $tpu_zone $tpu_project $tpu_gen $3 $4
    elif [ "$2" = "cp_ssh_key" ]; then
        _tpu_cp_ssh_key $tpu_zone $tpu_project $3
    elif [ "$2" = "retry_create" ]; then
        _tpu_retry_create $tpu_zone $tpu_project $tpu_gen $3 $4
    elif [ "$2" = "cs" ]; then
        _tpu_create $tpu_zone $tpu_project $tpu_gen $3 $4
        sleep 90s
        _tpu_setup $tpu_zone $tpu_project $4
    elif [ "$2" = "check" ]; then
        _tpu_check $tpu_zone $tpu_project $3
    elif [ "$2" = "setup" ]; then
        _tpu_setup $tpu_zone $tpu_project $3
    elif [ "$2" = "copy" ]; then
        _tpu_copy $tpu_zone $tpu_project $3
    elif [ "$2" = "stop" ]; then
        _tpu_stop $tpu_zone $tpu_project $3
    elif [ "$2" = "launch" ]; then
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "cl" ]; then
        _tpu_copy $tpu_zone $tpu_project $3
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "maintain" ]; then
        _tpu_maintain $tpu_zone $tpu_project $3
    elif [ "$2" = "ssh" ]; then
        _tpu_ssh $tpu_zone $tpu_project $3 "$4"
    elif [ "$2" = "reboot" ]; then
        _tpu_reboot $tpu_zone $tpu_project $3
    elif [ "$2" = "df" ]; then
        _tpu_ssh $tpu_zone $tpu_project $3 'df -h | grep root'
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi
    trap - SIGINT SIGTERM
}


export -f tpu _tpu_ips _tpu_create _tpu_setup _tpu_check _tpu_copy _tpu_stop _tpu_launch _tpu_maintain _tpu_ssh _tpu_reboot