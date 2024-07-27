cfg='fa\faf'

IFS=$'\\' read -ra foo <<< "$cfg"

# get the first element
a="${foo[-2]}"
echo $a