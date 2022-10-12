N,K = map(int,input().split())
board = [list(map(int,input().split())) for _ in range(N)]
horse_board = [[[-1]*3 for _ in range(N)] for _ in range(N)]
dirs = [[0,1],[0,-1],[-1,0],[1,0]]
horses = [list(map(int,input().split()))+[0] for _ in range(K)] # r,c,direction, linked list
for i in range(len(horses)):
    x,y,d,depth = horses[i]
    horses[i] = [x-1,y-1,d-1,depth]
    horse_board[x-1][y-1][0] = i

for t in range(1000):
    for idx in range(len(horses)):
        x,y,d,depth = horses[idx]
        dx,dy = dirs[d]
        nx,ny = x+dx,y+dy

        if not (0<=nx<N and 0<=ny<N) or board[nx][ny]==2: # Blue or out
            if d in [0,1]:
                d = 1-d
            else:
                d = 2+(3-d)
            horses[idx] = [x,y,d,depth]
            dx,dy = dirs[d]
            nx,ny = x+dx,y+dy

        # if t < 3:
        #     print(horses)

        if 0<=nx<N and 0<=ny<N and board[nx][ny]!=2:
            if board[nx][ny]==0: # white
                next_depth = 0
                while next_depth < 3 and horse_board[nx][ny][next_depth]!=-1:
                    next_depth += 1
                for i in range(depth, 3):
                    if horse_board[x][y][i] != -1:
                        if next_depth == 3: # End game
                            print(t+1)
                            exit()
                        horse_board[nx][ny][next_depth] = horse_board[x][y][i]
                        horses[horse_board[x][y][i]] = [nx,ny,horses[horse_board[x][y][i]][2],next_depth]
                        next_depth += 1
                        horse_board[x][y][i] = -1
                    else:
                        break

            elif board[nx][ny]==1: # red
                next_depth = 0
                while next_depth < 3 and horse_board[nx][ny][next_depth]!=-1:
                    next_depth += 1
                max_depth = depth
                while max_depth < 3 and horse_board[x][y][max_depth]!=-1:
                    max_depth += 1
                for i in reversed(range(depth, max_depth)):
                    if horse_board[x][y][i] != -1:
                        if next_depth == 3: # End game
                            print(t+1)
                            exit()
                        horse_board[nx][ny][next_depth] = horse_board[x][y][i]
                        horses[horse_board[x][y][i]] = [nx,ny,horses[horse_board[x][y][i]][2],next_depth]
                        next_depth += 1
                        horse_board[x][y][i] = -1
                    else:
                        break
                    

print(-1)