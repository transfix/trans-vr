#include <Socket/CVCSocket.h>
//************************************************************
// TCP routines
//************************************************************
CVCSocketTCP::CVCSocketTCP(int flag, int val, int _maxDataSizeByte) {
  m_maxDataSizeInByte = 0;
  m_buffer = NULL;
  m_bufferSize = 0;

  m_hasSubtree = false;
  m_socketBinded = false;
  m_connected = false;

  switch (flag) {
  case SERVER: {
    m_port = val;

    m_socket_id = socket(AF_INET, SOCK_STREAM, 0);
    if (m_socket_id == -1)
      fprintf(stderr, "socket generation fail\n");
    else {
      struct sockaddr_in serverAddr;
      serverAddr.sin_family = AF_INET;
      serverAddr.sin_port = htons(m_port);
      serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

      int ntry = 0;
      int bindret = -1;
      while (((bindret = bind(m_socket_id, (sockaddr *)&serverAddr,
                              sizeof(serverAddr))) == -1) &&
             (ntry++ < 10))
        ;

      if (bindret == -1)
        fprintf(stderr, "bind fail\n");
      else {
        fprintf(stderr, "binded[%d trial]\n", ntry + 1);
        m_socketBinded = true;
      }
    }
    m_IAM = SERVER;
    break;
  }
  case CLIENT: {
    m_nServer = val;
    m_socket_idv = new int[m_nServer];
    bool socketGenSuccess = true;
    for (int i = 0; i < m_nServer; i++) {
      m_socket_idv[i] = socket(AF_INET, SOCK_STREAM, 0);
      if (m_socket_idv[i] == -1) {
        fprintf(stderr, "socket generation fail(%d)\n", i);
        socketGenSuccess = false;
      }
    }

    if (socketGenSuccess)
      m_socketBinded = true;

    m_IAM = CLIENT;
    break;
  }
  default:
    fprintf(stderr, "SOCKET type error\n");
  }

  m_maxDataSizeInByte = _maxDataSizeByte;
  m_bufferSize = m_maxDataSizeInByte + sizeof(short int) + sizeof(short int);
  m_buffer =
      new char[m_bufferSize]; // Additional 4 bytes for flag and data size
}

CVCSocketTCP::~CVCSocketTCP() {
  if (m_socketBinded) {
    if (m_buffer)
      delete[] m_buffer;
    if (shutdown(m_socket_id, 2) == -1)
      fprintf(stderr, "socket closing fail\n");
  }
  m_targetAddr.clear();
}

bool CVCSocketTCP::_connect(void) {
  if (m_connected)
    return true;

  switch (m_IAM) {
  case SERVER: {
    fprintf(stderr, "start to listen\n");
    int ntry = 0;
    while (!m_connected && ntry++ < 20) {
      if (listen(m_socket_id, BACKLOG) == -1) {
        fprintf(stderr, "can not listen\n");
        continue;
      } else if ((m_socket_id = accept(m_socket_id, (sockaddr *)&m_clientAddr,
                                       &m_clientLen)) == -1) {
        fprintf(stderr, "can not accept\n");
        continue;
      } else {
        fprintf(stderr, "connected[sockid: %d, try: %d]\n", m_socket_id,
                ntry + 1);
        m_connected = true;
      }
    }
  } break;
  case CLIENT: {
    fprintf(stderr, "start to connect\n");
    bool connectSuccess = false;
    m_connectedv = new bool[m_nServer];
    for (int i = 0; i < m_nServer; i++)
      m_connectedv[i] = false;

    int ntry = 0;
    while (!connectSuccess && ntry++ < 20) {
      connectSuccess = true;
      for (int i = 0; i < m_nServer; i++) {
        if (m_connectedv[i])
          continue;
        if (connect(m_socket_idv[i], (sockaddr *)&(m_targetAddr[i]),
                    sizeof(m_targetAddr[i])) == -1) {
          fprintf(stderr, "can not connect[socket id: %d]\n",
                  m_socket_idv[i]);
          connectSuccess = false;
        } else
          m_connectedv[i] = true;
      }
    }
    if (connectSuccess) {
      m_connected = true;
      fprintf(stderr, "connected\n");
    } else
      return false;
  } break;
  default:
    fprintf(stderr, "SOCKET type error\n");
    return false;
  }
  return true;
}

bool CVCSocketTCP::_receive(short int *_flag, void *_data, int *_dataSize) {
  if (!m_connected)
    return false;

  sockaddr_in clientAddr;
  socklen_t clientLen;

  short int recved = 0;
  if ((recved = recv(m_socket_id, m_buffer, m_bufferSize, 0)) == -1) {
    fprintf(stderr, "recvfrom fail\n");
    return false;
  }
  short int flag = 0;
  short int size = 0;
  memcpy(&flag, m_buffer, sizeof(short int));
  memcpy(&size, m_buffer + sizeof(short int), sizeof(short int));
  if (recved != size) {
    fprintf(stderr, "full data receiving fail\n");
    return false;
  }

  *_dataSize = recved - 2 * sizeof(short int);
  memcpy(_data, m_buffer + 2 * sizeof(short int), *_dataSize);
  *_flag = flag;

  return true;
}

bool CVCSocketTCP::_wreceive(short int *_flag, void *_data, int *_dataSize) {
  if (!m_connected)
    return false;

  sockaddr_in clientAddr;
  socklen_t clientLen;

  short int recved = 0;
#ifdef DEBUG_SOCKET
  fprintf(stderr, "socketid: %d\n", m_socket_id);
#endif

  bzero(m_buffer, m_bufferSize);
  if ((recved = read(m_socket_id, (void *)m_buffer, m_bufferSize)) == -1) {
    fprintf(stderr, "socket read fail\n");
    return false;
  }

  short int flag = 0;
  short int size = 0;
  memcpy(&flag, m_buffer, sizeof(short int));
  memcpy(&size, m_buffer + sizeof(short int), sizeof(short int));
  if (recved < size) {
    fprintf(stderr, "full data receiving fail: recved:%d, size: %d\n", recved,
            size);
    return false;
  }
#ifdef DEBUG_SOCKET
  fprintf(stderr, "full data receiving fail: recved:%d, size: %d\n", recved,
          size);
#endif

  *_dataSize = size - 2 * sizeof(short int);
  bzero(_data, 1024);
  memcpy(_data, m_buffer + 2 * sizeof(short int), *_dataSize);
  *_flag = flag;
#ifdef DEBUG_SOCKET
  fprintf(stderr, "%s\n\n", (char *)_data);
#endif
  return true;
}

bool CVCSocketTCP::_wreceive(std::vector<MSG> *_msg) {
  if (!m_connected)
    return false;

  sockaddr_in clientAddr;
  socklen_t clientLen;

  short int recved = 0;
#ifdef DEBUG_SOCKET
  fprintf(stderr, "socketid: %d\n", m_socket_id);
#endif
  bzero(m_buffer, m_bufferSize);
  if ((recved = read(m_socket_id, (void *)m_buffer, m_bufferSize)) == -1) {
    fprintf(stderr, "socket read fail\n");
    return false;
  }

  int checkBufSize = m_bufferSize;
  char *bufptr = m_buffer;
  while (1) {
#ifdef DEBUG_SOCKET
    fprintf(stderr, "max data size: %d\n", m_maxDataSizeInByte);
#endif

    MSG msg;
    short int size = 0;
    memcpy(&(msg.flag), bufptr, sizeof(short int));
    bufptr += sizeof(short int);
    memcpy(&size, bufptr, sizeof(short int));
    bufptr += sizeof(short int);

    if (size > checkBufSize) {
      fprintf(stderr,
              "received data is larger than maximum buffer size[buf: %d, "
              "data: %d]\n",
              checkBufSize, size);
      break;
    }

    msg.size = size - 2 * sizeof(short int);

    bzero(msg.data, 1024);
    memcpy(msg.data, bufptr, msg.size);

#ifdef DEBUG_SOCKET
    fprintf(stderr, "recved: %d, size: %d, data: %s\n\n", recved, size,
            msg.data);
#endif

    _msg->push_back(msg);

    recved -= size;
    checkBufSize -= size;
    if (recved > 0)
      bufptr += msg.size;
    else
      break;
  }
#ifdef DEBUG_SOCKET
  fprintf(stderr, "size of msg: %d\n", _msg->size());
#endif
  return true;
}

bool CVCSocketTCP::_send(short int _flag, const void *_data, int _dataSize) {
  memcpy(m_buffer, &_flag, sizeof(short int));
  short int fullSize = _dataSize + 2 * sizeof(short int);
  if (m_bufferSize < fullSize) {
    fprintf(stderr, "data size exceed buffer size\n");
    return false;
  }

  memcpy(m_buffer + sizeof(short int), &fullSize, sizeof(short int));
  memcpy(m_buffer + 2 * sizeof(short int), _data, _dataSize);

#ifdef DEBUG_SOCKET
  fprintf(stderr, "sending %d bytes\n", fullSize);
#endif

  bool hasFail = false;
  for (int i = 0; i < m_targetAddr.size(); i++) {
    if (!m_connectedv[i])
      continue;
    short int sent = 0;
    // if( (sent = send( m_socket_idv[i], m_buffer, fullSize, 0 )) == -1 ) {
    if ((sent = write(m_socket_idv[i], m_buffer, fullSize)) == -1) {
      fprintf(stderr, "target[%d] sendto fail\n", i);
      hasFail = true;
    } else if (sent != fullSize) {
      fprintf(stderr, "target[%d] full data sending fail\n", i);
      hasFail = true;
    }
  }
  return !hasFail;
}

bool CVCSocketTCP::setTargets(char *_fname) {
  m_targetAddr.clear();

  FILE *fp = fopen(_fname, "r");
  if (!fp) {
    fprintf(stderr, "%s file open fail\n", _fname);
    return false;
  }
  char line[1025];
  fgets(line, 1024, fp);
  int w, h;
  sscanf(line, "%d %d", &w, &h);
  for (int i = 0; i < w * h; i++) {
    if (fgets(line, 1024, fp) == NULL) {
      fprintf(stderr, "wrong server list file\n");
      break;
    }
    char hostname[256] = {0};
    int port = 0;
    sscanf(line, "%s %d", hostname, &port);
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    struct hostent *record = gethostbyname(hostname);
    memcpy(&(serverAddr.sin_addr), record->h_addr, record->h_length);
    m_targetAddr.push_back(serverAddr);
  }
  fclose(fp);
  return true;
}

void CVCSocketTCP::setTargets(int _nHosts, std::string *_hostnames,
                              int *_ports) {
  m_targetAddr.clear();

  for (int i = 0; i < _nHosts; i++) {
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(_ports[i]);
    struct hostent *record = gethostbyname(_hostnames[i].data());
    memcpy(&(serverAddr.sin_addr), record->h_addr, record->h_length);
    m_targetAddr.push_back(serverAddr);
  }
}

//************************************************************
// UDP routines
//************************************************************
CVCSocketUDP::CVCSocketUDP(int _port, int _maxDataSizeByte) {
  m_maxDataSizeInByte = 0;
  m_buffer = NULL;
  m_bufferSize = 0;

  m_hasSubtree = false;
  m_socketBinded = false;
  m_port = _port;

  m_socket_id = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (m_socket_id == -1)
    fprintf(stderr, "socket generation fail\n");
  else {
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(_port);
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(m_socket_id, (sockaddr *)&serverAddr, sizeof(serverAddr)) == -1)
      fprintf(stderr, "bind fail\n");
    else {
      m_maxDataSizeInByte = _maxDataSizeByte;
      m_buffer = new char[m_maxDataSizeInByte + sizeof(short int) +
                          sizeof(short int)]; // Additional 4 bytes for flag
                                              // and data size

      m_socketBinded = true;
    }
  }
}

CVCSocketUDP::~CVCSocketUDP() {
  if (m_socketBinded) {
    if (m_buffer)
      delete[] m_buffer;
    if (shutdown(m_socket_id, 2) == -1)
      fprintf(stderr, "socket closing fail\n");
  }
  m_targetAddr.clear();
}

bool CVCSocketUDP::receive(short int *_flag, void *_data, int *_dataSize) {
  if (!m_socketBinded)
    return false;

  sockaddr_in clientAddr;
  socklen_t clientLen;

  short int recved = 0;
  if ((recved = recvfrom(m_socket_id, m_buffer, sizeof(m_buffer), 0,
                         (sockaddr *)&clientAddr, &clientLen)) == -1) {
    fprintf(stderr, "recvfrom fail\n");
    return false;
  }
  short int flag = 0;
  short int size = 0;
  memcpy(&flag, m_buffer, sizeof(short int));
  memcpy(&size, m_buffer + sizeof(short int), sizeof(short int));
  if (recved != size) {
    fprintf(stderr, "full data receiving fail\n");
    return false;
  }

  *_dataSize = recved - 2 * sizeof(short int);
  memcpy(_data, m_buffer + 2 * sizeof(short int), *_dataSize);
  *_flag = flag;

  return true;
}

bool CVCSocketUDP::wreceive(short int *_flag, void *_data, int *_dataSize) {
  if (!m_socketBinded)
    return false;

  sockaddr_in clientAddr;
  socklen_t clientLen;

  short int recved = 0;
  short int flag = 0;
  short int size = 0;
  while (1) {
    fprintf(stderr, "in while\n");
    if ((recved = recvfrom(m_socket_id, m_buffer, sizeof(m_buffer), 0,
                           (sockaddr *)&clientAddr, &clientLen)) == -1) {
      fprintf(stderr, "recv fail\n");
      continue;
    }

    memcpy(&size, m_buffer + sizeof(short int), sizeof(short int));
    if (recved == size)
      break;
    else
      fprintf(stderr, "recved: %d, size: %d\n", recved, size);
  }
  memcpy(&flag, m_buffer, sizeof(short int));

  *_dataSize = recved - 2 * sizeof(short int);
  memcpy(_data, m_buffer + 2 * sizeof(short int), *_dataSize);
  *_flag = flag;

  return true;
}

bool CVCSocketUDP::send(short int _flag, const void *_data, int _dataSize) {
  memcpy(m_buffer, &_flag, sizeof(short int));
  short int fullSize = _dataSize + 2 * sizeof(short int);
  memcpy(m_buffer + sizeof(short int), &fullSize, sizeof(short int));
  memcpy(m_buffer + 2 * sizeof(short int), _data, _dataSize);

  bool hasFail = false;
  for (int i = 0; i < m_targetAddr.size(); i++) {
    short int sent = 0;
    if ((sent = sendto(m_socket_id, m_buffer, fullSize, 0,
                       (sockaddr *)&(m_targetAddr[i]),
                       sizeof(m_targetAddr[i]))) == -1) {
      fprintf(stderr, "target[%d] sendto fail\n", i);
      hasFail = true;
    } else if (sent != fullSize) {
      fprintf(stderr, "target[%d] full data sending fail\n", i);
      hasFail = true;
    }
  }
  return !hasFail;
}

bool CVCSocketUDP::setTargets(char *_fname) {
  m_targetAddr.clear();

  FILE *fp = fopen(_fname, "r");
  if (!fp) {
    fprintf(stderr, "%s file open fail\n", _fname);
    return false;
  }
  char line[1025];
  fgets(line, 1024, fp);
  int w, h;
  sscanf(line, "%d %d", &w, &h);
  for (int i = 0; i < w * h; i++) {
    if (fgets(line, 1024, fp) == NULL) {
      fprintf(stderr, "wrong server list file\n");
      break;
    }
    char hostname[256] = {0};
    int port = 0;
    sscanf(line, "%s %d", hostname, &port);
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    struct hostent *record = gethostbyname(hostname);
    memcpy(&(serverAddr.sin_addr), record->h_addr, record->h_length);
    m_targetAddr.push_back(serverAddr);
  }
  fclose(fp);
  return true;
}

void CVCSocketUDP::setTargets(int _nHosts, std::string *_hostnames,
                              int *_ports) {
  m_targetAddr.clear();

  for (int i = 0; i < _nHosts; i++) {
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(_ports[i]);
    struct hostent *record = gethostbyname(_hostnames[i].data());
    memcpy(&(serverAddr.sin_addr), record->h_addr, record->h_length);
    m_targetAddr.push_back(serverAddr);
  }
}
