#!/bin/bash

if [ "$1" = "local" ]; then
	echo "Switching to hotspot mode"

	sudo nmcli connection down eduroam 2>/dev/null
	sudo nmcli connection up pi-hotspot

elif [ "$1" = "eduroam" ]; then
	echo "Switching to eduroam"

	sudo nmcli connection down pi-hotspot 2>/dev/null
	sudo nmcli connection up eduroam

else
	echo "Usage: $0 {local|eduroam}"
fi
