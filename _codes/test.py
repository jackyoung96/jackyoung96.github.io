from itertools import combinations, product

result = []

def dfs(teams_idx, res_tab):
    if teams_idx == len(teams):
        if any(res_tab):
            return 0
        else:
            return 1

    a,b = teams[teams_idx]
    for game in range(3):
        if res_tab[3*a+game] == 0:
            continue
        if res_tab[3*b+(2-game)] == 0:
            continue
        res_copy = [i for i in res_tab]
        res_copy[3*a+game] -= 1
        res_copy[3*b+(2-game)] -= 1
        
        res = dfs(teams_idx+1, res_copy)
        if res == 1:
            return res
    return 0

for _ in range(4):
    res = list(map(int, input().split()))

    teams = list(combinations(range(6),2))
    res_copy = [i for i in res]

    ans = dfs(0, res_copy)
    result.append(ans)

print(*result)