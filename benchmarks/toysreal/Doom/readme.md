Doom requires these packages

$ sudo apt-get install libsdl-mixer1.2
$ sudo apt-get install libsdl-net1.2
$ sudo apt-get install libpng12

The last one might be harder to get since newer Linux distributions use libpng16.
Luckly, I found this lib within other softwares by running

$ locate libpng12

Then, make a symbolic link to 

$ sudo ln -s <path>/libpng12.so.0 /usr/lib/x86_64-linux-gnu/libpng12.so.0

