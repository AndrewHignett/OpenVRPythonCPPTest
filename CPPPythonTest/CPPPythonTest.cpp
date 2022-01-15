#include <iostream>
#include <WS2tcpip.h>
#include "Connection.h"
#include "Parameters.h"
#include "Tracker.h"

#include <vector>
#include <sstream>

// added openvr_api.dll to debug file path

// Include the Winsock library (lib) file
#pragma comment (lib, "ws2_32.lib")

// Saves us from typing std::cout << etc. etc. etc.
using namespace std;

// Main entry point into the server
void main()
{
	////////////////////////////////////////////////////////////
	// INITIALIZE WINSOCK
	////////////////////////////////////////////////////////////

	// Structure to store the WinSock version. This is filled in
	// on the call to WSAStartup()
	WSADATA data;

	// To start WinSock, the required version must be passed to
	// WSAStartup(). This server is going to use WinSock version
	// 2 so I create a word that will store 2 and 2 in hex i.e.
	// 0x0202
	WORD version = MAKEWORD(2, 2);

	// Start WinSock
	int wsOk = WSAStartup(version, &data);
	if (wsOk != 0)
	{
		// Not ok! Get out quickly
		cout << "Can't start Winsock! " << wsOk;
		return;
	}

	////////////////////////////////////////////////////////////
	// SOCKET CREATION AND BINDING
	////////////////////////////////////////////////////////////

	// Create a socket, notice that it is a user datagram socket (UDP)
	SOCKET in = socket(AF_INET, SOCK_DGRAM, 0);

	// Create a server hint structure for the server
	sockaddr_in serverHint;
	serverHint.sin_addr.S_un.S_addr = ADDR_ANY; // Us any IP address available on the machine
	serverHint.sin_family = AF_INET; // Address format is IPv4
	serverHint.sin_port = htons(8888); // Convert from little to big endian

	// Try and bind the socket to the IP and port
	//bind(in, (sockaddr*)&serverHint, sizeof(serverHint));
	//including <thread> causes issues here
	if (::bind(in, (sockaddr*)&serverHint, sizeof(serverHint)) == SOCKET_ERROR)
	{
		cout << "Can't bind socket! " << WSAGetLastError() << endl;
		return;
	}
	

	////////////////////////////////////////////////////////////
	// MAIN LOOP SETUP AND ENTRY
	////////////////////////////////////////////////////////////

	sockaddr_in client; // Use to hold the client information (port / ip address)
	int clientLength = sizeof(client); // The size of the client information

	char buf[1024];
	
	//test
	Parameters *params = new Parameters();
	Connection *con = new Connection(params);
	Tracker *tracker = new Tracker(params, con);


	// Enter a loop
	while (true)
	{
		ZeroMemory(&client, clientLength); // Clear the client structure
		ZeroMemory(buf, 1024); // Clear the receive buffer

		// Wait for message
		int bytesIn = recvfrom(in, buf, 1024, 0, (sockaddr*)&client, &clientLength);
		if (bytesIn == SOCKET_ERROR)
		{
			cout << "Error receiving from client " << WSAGetLastError() << endl;
			continue;
		}

		// Display message and client info
		char clientIp[256]; // Create enough space to convert the address byte array
		ZeroMemory(clientIp, 256); // to string of characters

		// Convert from byte array to chars
		inet_ntop(AF_INET, &client.sin_addr, clientIp, 256);

		// Display the message / who sent it
		cout << "Message recv from " << clientIp << " : " << buf <<  endl;

		switch (buf[0])
		{
			//these must be done in this order
			case '+':
				//calibrate
				break;
			case '%':
				//connect
				con->StartConnection();
				break;
			case '#':
				//start trackers
				tracker->Start();
				break;
			default:
				//set capture data to trackers
				double ax, ay, az, bx, by, bz, cx, cy, cz;
				//will need to use other coordinates to calibrate, scale and get direction (need to understand quaternions)
				vector<string> result;
				stringstream s_stream(buf); //create string stream from the string
				while (s_stream.good()) {
					string substr;
					getline(s_stream, substr, ','); //get first string delimited by comma
					result.push_back(substr);
				}
				for (int i = 0; i < result.size(); i++) {    //print all splitted strings
					result.at(i);
				}
				ax = stod(result.at(0));
				ay = stod(result.at(1));
				az = stod(result.at(2));
				bx = stod(result.at(3));
				by = stod(result.at(4));
				bz = stod(result.at(5));
				cx = stod(result.at(6));
				cy = stod(result.at(7));
				cz = stod(result.at(8));
				tracker->testFunction(ax, ay, az, bx, by, bz, cx, cy, cz);
				break;
		}
	}

	// Close socket
	closesocket(in);

	// Shutdown winsock
	WSACleanup();
}


//use hand positions, this could be used for scaling coordinate space and dealing with relative locations
//This would be easy if moving close to and further away from the camera makes no difference to the coorindates

//Track direction the user is facing, this should be useable for foot tracker direction