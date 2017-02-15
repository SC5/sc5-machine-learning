#! /bin/bash

vw --cb_explore 4 --daemon --port 26542 --epsilon 0.1 && tail -f /dev/stdout