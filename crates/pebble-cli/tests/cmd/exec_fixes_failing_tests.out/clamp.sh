clamp() {
    local value=$1 lower=$2 upper=$3
    if (( value < lower )); then
        value=$lower
    elif (( value > upper )); then
        value=$upper
    fi
    echo "$value"
}
