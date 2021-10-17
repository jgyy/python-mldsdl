"""
Cleaning Your Data
"""
from os.path import join, dirname
from re import compile as recompile


def url_count_val(url, url_counts):
    """
    url check function
    """
    if url in url_counts:
        url_counts[url] = url_counts[url] + 1
    else:
        url_counts[url] = 1
    return url_counts


def url_check_1():
    """
    url check 1 function
    """
    url_counts = {}
    with open(LOG_PATH, "r", encoding="utf-8") as file:
        for line in (l.rstrip() for l in file):
            match = FORMAT_PAT.match(line)
            if match:
                access = match.groupdict()
                request = access["request"]
                if len(request.split()) == 3:
                    _, url, _ = request.split()
                    url_counts = url_count_val(url, url_counts)
    results = sorted(url_counts, key=lambda i: int(url_counts[i]), reverse=True)
    for result in results[:20]:
        print(result + ": " + str(url_counts[result]))


def url_check_2():
    """
    url check 2 function
    """
    with open(LOG_PATH, "r", encoding="utf-8") as file:
        for line in (l.rstrip() for l in file):
            match = FORMAT_PAT.match(line)
            if match:
                access = match.groupdict()
                request = access["request"]
                fields = request.split()
                if len(fields) != 3:
                    print(fields)


def url_check_3():
    """
    url check 3 function
    """
    url_counts = {}
    with open(LOG_PATH, "r", encoding="utf-8") as file:
        for line in (l.rstrip() for l in file):
            match = FORMAT_PAT.match(line)
            if match:
                access = match.groupdict()
                request = access["request"]
                fields = request.split()
                if len(fields) == 3:
                    url = fields[1]
                    url_counts = url_count_val(url, url_counts)
    results = sorted(url_counts, key=lambda i: int(url_counts[i]), reverse=True)
    for result in results[:20]:
        print(result + ": " + str(url_counts[result]))


def url_check_4():
    """
    url check 4 function
    """
    url_counts = {}
    with open(LOG_PATH, "r", encoding="utf-8") as file:
        for line in (l.rstrip() for l in file):
            match = FORMAT_PAT.match(line)
            if match:
                access = match.groupdict()
                request = access["request"]
                fields = request.split()
                if len(fields) == 3:
                    action, url, _ = fields
                    if action == "GET":
                        url_counts = url_count_val(url, url_counts)
    results = sorted(url_counts, key=lambda i: int(url_counts[i]), reverse=True)
    for result in results[:20]:
        print(result + ": " + str(url_counts[result]))


def wrapper():
    """
    wrapper function
    """
    url_check_1()
    url_check_2()
    url_check_3()
    url_check_4()

    user_agents = {}
    with open(LOG_PATH, "r", encoding="utf-8") as file:
        for line in (l.rstrip() for l in file):
            match = FORMAT_PAT.match(line)
            if match:
                access = match.groupdict()
                agent = access["user_agent"]
                if agent in user_agents:
                    user_agents[agent] = user_agents[agent] + 1
                else:
                    user_agents[agent] = 1
    results = sorted(user_agents, key=lambda i: int(user_agents[i]), reverse=True)
    for result in results:
        print(result + ": " + str(user_agents[result]))

    url_counts = {}
    with open(LOG_PATH, "r", encoding="utf-8") as file:
        for line in (l.rstrip() for l in file):
            match = FORMAT_PAT.match(line)
            if match:
                access = match.groupdict()
                agent = access["user_agent"]
                if not (
                    "bot" in agent
                    or "spider" in agent
                    or "Bot" in agent
                    or "Spider" in agent
                    or "W3 Total Cache" in agent
                    or agent == "-"
                ):
                    request = access["request"]
                    fields = request.split()
                    if len(fields) == 3:
                        action, url, _ = fields
                        if action == "GET":
                            url_counts = url_count_val(url, url_counts)
    results = sorted(url_counts, key=lambda i: int(url_counts[i]), reverse=True)
    for result in results[:20]:
        print(result + ": " + str(url_counts[result]))


if __name__ == "__main__":
    FORMAT_PAT = recompile(
        r"(?P<host>[\d\.]+)\s"
        r"(?P<identity>\S*)\s"
        r"(?P<user>\S*)\s"
        r"\[(?P<time>.*?)\]\s"
        r'"(?P<request>.*?)"\s'
        r"(?P<status>\d+)\s"
        r"(?P<bytes>\S*)\s"
        r'"(?P<referer>.*?)"\s'
        r'"(?P<user_agent>.*?)"\s*'
    )
    LOG_PATH = join(dirname(__file__), "access_log.txt")
    wrapper()
