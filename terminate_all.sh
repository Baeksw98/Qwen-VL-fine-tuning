#!/bin/bash

# Check if sudo is available
if ! command -v sudo &> /dev/null
then
    echo "sudo is not installed. Please install sudo or run this script as root."
    exit 1
fi

# Find all Python processes (both active and defunct)
python_pids=$(ps aux | grep python | grep -v grep | awk '{print $2}')

if [ -z "$python_pids" ]; then
    echo "No Python processes found."
    exit 0
fi

echo "Found the following Python process PIDs:"
echo "$python_pids"

read -p "Do you want to forcefully terminate all these processes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    for pid in $python_pids
    do
        echo "Attempting to forcefully terminate process $pid"
        sudo kill -9 $pid 2>/dev/null
    done
    echo "Attempted to terminate all Python processes."
else
    echo "No processes were terminated."
fi

echo "Remaining Python processes:"
sudo ps aux | grep python | grep -v grep