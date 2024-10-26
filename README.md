# FedL-Transformer_Wireless
Fedreated Transformer over Wireless Communication

Enable packet loss in OpenWRT
```shell
tc qdisc add dev wlan1 root netem loss 5%
tc qdisc show dev wlan1
```

Remove packet loss
```shell
tc qdisc del dev wlan1 root
```

### Data encoding and transfer
* Original 59 numpy arrays
* Convert all of them to the bitstream. There are 59 bitstreams.
* Encode each of the arrays: 1024 bits, encode, rsvl, awgn. Rate = 0.5. Add an fixed irrelevent  bits to the final codeWord to make it 1024bits. Each array has $M_i$ codeWord
* Total number of the codeWord $N=\sum M_i$. Each codeWord has 1024 bits.
* Create UPD packets. Each UDP packet has $(2^n * N + 32)$ bits. $32$ is the length of the codeWord's series number. In my case, $N=1100$
* When packet lose happened, each codeWord will lose $2^n$ bits