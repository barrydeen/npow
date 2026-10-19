// Minimal HTTP/1.1 server for the npow JSON API. Thread-per-connection,
// CORS enabled, Connection: close. Header-only; no external dependencies.
#pragma once

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <thread>

struct HttpRequest {
  std::string method, path, body;
};

struct HttpResponse {
  int status = 200;
  std::string content_type = "application/json";
  std::string body;
};

// Returns the listen fd so the caller can shutdown() it on SIGINT.
inline int http_listen(const std::string& bind_addr, int port) {
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) return -1;
  int one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);
  addr.sin_addr.s_addr =
      bind_addr.empty() || bind_addr == "0.0.0.0" ? INADDR_ANY
                                                  : inet_addr(bind_addr.c_str());
  if (::bind(srv, (sockaddr*)&addr, sizeof addr) < 0) {
    close(srv);
    return -1;
  }
  if (listen(srv, 16) < 0) {
    close(srv);
    return -1;
  }
  return srv;
}

// Blocking accept loop; spawns a detached thread per connection.
// Returns when stop_flag is set (or on accept error).
inline void http_serve(int srv, const std::atomic<bool>& stop_flag,
                       const std::function<HttpResponse(const HttpRequest&)>&
                           handler) {
  while (!stop_flag.load()) {
    sockaddr_in cli{};
    socklen_t cl = sizeof cli;
    int fd = accept(srv, (sockaddr*)&cli, &cl);
    if (fd < 0) {
      if (errno == EINTR || errno == EBADF) break;
      continue;
    }
    std::thread([fd, handler]() {
      std::string buf;
      char tmp[8192];
      size_t header_end = std::string::npos;
      while (header_end == std::string::npos) {
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) {
          close(fd);
          return;
        }
        buf.append(tmp, n);
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > 4 * 1024 * 1024) {
          close(fd);
          return;
        }
      }
      std::string headers = buf.substr(0, header_end);
      std::string rest = buf.substr(header_end + 4);
      HttpRequest req;
      size_t content_length = 0;
      {
        std::istringstream is(headers);
        std::string line;
        std::getline(is, line);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t sp1 = line.find(' ');
        size_t sp2 = line.rfind(' ');
        if (sp1 != std::string::npos && sp2 != std::string::npos &&
            sp2 > sp1) {
          req.method = line.substr(0, sp1);
          req.path = line.substr(sp1 + 1, sp2 - sp1 - 1);
        }
        while (std::getline(is, line)) {
          if (!line.empty() && line.back() == '\r') line.pop_back();
          size_t colon = line.find(':');
          if (colon == std::string::npos) continue;
          std::string key = line.substr(0, colon);
          std::string val = line.substr(colon + 1);
          for (auto& c : key) c = (char)tolower((unsigned char)c);
          // trim leading space
          size_t b = val.find_first_not_of(" \t");
          if (b != std::string::npos) val = val.substr(b);
          if (key == "content-length") content_length = strtoull(val.c_str(), nullptr, 10);
        }
      }
      if (content_length > 2 * 1024 * 1024) {
        close(fd);
        return;
      }
      while (rest.size() < content_length) {
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) {
          close(fd);
          return;
        }
        rest.append(tmp, n);
      }
      req.body = rest.substr(0, content_length);

      HttpResponse resp = handler(req);
      const char* status_text = resp.status == 200 ? "OK"
                                : resp.status == 204 ? "No Content"
                                : resp.status == 400 ? "Bad Request"
                                : resp.status == 404 ? "Not Found"
                                : resp.status == 409 ? "Conflict"
                                : resp.status == 500 ? "Internal Server Error"
                                                     : "OK";
      std::string out = "HTTP/1.1 " + std::to_string(resp.status) + " " +
                        status_text + "\r\n"
                        "Content-Type: " + resp.content_type + "\r\n"
                        "Content-Length: " +
                        std::to_string(resp.body.size()) + "\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: Content-Type\r\n"
                        "Access-Control-Max-Age: 86400\r\n"
                        "Connection: close\r\n\r\n";
      out += resp.body;
      size_t off = 0;
      while (off < out.size()) {
        ssize_t n = send(fd, out.data() + off, out.size() - off, MSG_NOSIGNAL);
        if (n <= 0) break;
        off += (size_t)n;
      }
      close(fd);
    }).detach();
  }
}
