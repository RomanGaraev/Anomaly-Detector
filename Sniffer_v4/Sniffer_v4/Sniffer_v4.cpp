// Sniffer_v4.cpp: определяет точку входа для консольного приложения.
//
#include "stdafx.h"
#include <iostream>
#include <tins/tins.h>

using namespace Tins;
using namespace std;

bool callback(const PDU &pdu) {
	// Find the IP layer
	const IP &ip = pdu.rfind_pdu<IP>();
	// Find the TCP layer
	const TCP &tcp = pdu.rfind_pdu<TCP>();
	cout << ip.src_addr() << ':' << tcp.sport() << " " << ip.size() << " -> "
		 << ip.dst_addr() << ':' << tcp.dport() << endl;
	return true;
}

int main() {
	SnifferConfiguration conf;
	NetworkInterface lo1(IPv4Address("192.168.1.105"));
	conf.set_promisc_mode(true);
	Sniffer(lo1.name(), conf).sniff_loop(callback);
}