#ifndef __WINDOWBUF_H__
#define __WINDOWBUF_H__

#include <iostream>
#include <cstdio>
#include <sstream>

#include <QCoreApplication>

#include <VolumeRover2/CVCMainWindow.h>

namespace CVC_NAMESPACE
{

class windowbuf : public std::streambuf {
public:
  windowbuf (CVCMainWindow* w) { 
    win = w; 
  }

  int sync ()
  { 
    std::streamsize n = pptr () - pbase ();
    return (n && write_to_window (win, pbase (), n) != n) ? EOF : 0;
  }

  int overflow (int ch)
  { 
    std::streamsize n = pptr () - pbase ();
    if (n && sync ())
      return EOF;
    if (ch != EOF)
    {
      char cbuf[1];
      cbuf[0] = ch;
      if (write_to_window (win, cbuf, 1) != 1)
        return EOF;
    }
    pbump (-n);  // Reset pptr().
    return 0;
  }

  std::streamsize xsputn (char* text, std::streamsize n)
  { return sync () == EOF ? 0 : write_to_window (win, text, n); }

private:
  int write_to_window (CVCMainWindow* win, char* text, int n)
  {
    if (n == 0) return 1;
    buf += std::string(text, n);
    // std::string s(text, n);
    // ss << s;
    if (n > 1 || text[0] == '\n') {
      typedef CVC_NAMESPACE::CVCMainWindow::CVCMainWindowEvent Event;
      // Event* e = new Event("logEntry", ss.str());
      Event* e = new Event("logEntry", buf);
      QCoreApplication::postEvent(win, e);
      // ss = std::stringstream();
      buf = std::string();
    }
    return 1;
  }

private:
  // std::stringstream ss;
  std::string buf;
  CVCMainWindow* win;
};

}

#endif
