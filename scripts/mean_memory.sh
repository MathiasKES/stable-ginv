#!/bin/bash

while read -r f; do [ -f "$f" ] && echo "$f"; done < ./scripts/get_results.txt | \
xargs -d '\n' awk '/Max Memory/{max+=$(NF-1); m++; if($(NF-1)>peak) peak=$(NF-1)} /Average Memory/{avg+=$(NF-1); a++} END{printf "Mean Max Memory: %.2f MB\nPeak Max Memory: %.2f MB\nMean Average Memory: %.2f MB\n", max/m, peak, avg/a}'