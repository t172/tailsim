#!/bin/bash

groups='tth ttjets ttW'

function plot_dir() {
    for c in $('ls' "$HOME/www/unit_scale/${g}/${n}"); do
	cd "$HOME/www/unit_scale/${g}/${n}/${c}/"
	for f in *.gp; do
	    echo "Plotting ${g}-${n}-${c}: $f"
	    gnuplot "$f"
	done
    done
}

for g in $groups; do
    echo "$g"
    for json in $('ls' $HOME/data/${g}/*.json); do
	n="$(basename ${json/.json/})"
	echo "$g-$n"
	resource_monitor_split -J "$json" -D "${json/.json/.db}" "$HOME/www/unit_scale/${g}/${n}"
	wait
	plot_dir &
    done
done
wait
