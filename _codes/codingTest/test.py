xs = [-1,-1,0,1,1,1,0,-1]
ys = [0,-1,-1,-1,0,1,1,1]

board = [[] for _ in range(4)]
fish_dict = {}
for i in range(4):
    fishes = list(map(int,input().split()))
    for j in range(4):
        board[i].append([fishes[2*j],fishes[2*j+1]-1])
        fish_dict[fishes[2*j]] = [i,j]

score = board[0][0][0]
fish_dict[board[0][0][0]] = [-1,-1]
board[0][0][0] = 17 # shark
fish_dict[17] = [0,0]

def move_fishes(fish_dict, board):
    for f_idx in range(1,17):
        if fish_dict[f_idx][0]==-1:
            continue
        
        x,y = fish_dict[f_idx]
        head = board[x][y][1]
        for _ in range(8):
            dx,dy = xs[head], ys[head]
            if 0<=x+dx<4 and 0<=y+dy<4 and board[x+dx][y+dy][0]!=17:
                nx,ny = x+dx, y+dy
                if board[nx][ny][0] != 0:
                    board[x][y] = [board[nx][ny][0], board[nx][ny][1]]
                    fish_dict[board[x][y][0]] = [x,y]
                else:
                    board[x][y] = [0,0]
                board[nx][ny] = [f_idx,head]
                fish_dict[f_idx] = [nx,ny]

                break
            head = (head + 1) % 8
        # print('------------')
        # print(*board,sep='\n')
    
    # print('------------')
    # print(*board,sep='\n')

def dfs(fish_dict, board, score):
    move_fishes(fish_dict, board)

    x,y = fish_dict[17]
    head = board[x][y][1]
    dx, dy = xs[head], ys[head]
    n = 1
    scores = []
    while 0<=x+n*dx<4 and 0<=y+n*dy<4:
        nx,ny = x+n*dx, y+n*dy
        if board[nx][ny][0] != 0 and board[nx][ny][0] != 17:
            new_fish_dict = {k:v for k,v in fish_dict.items()}
            new_board = [[i for i in line] for line in board]
            fish = new_board[nx][ny][0]
            new_fish_dict[fish]=[-1,-1]
            new_board[nx][ny][0] = 17
            new_board[x][y] = [0,0]
            new_fish_dict[17] = [nx,ny]
            scores.append(dfs(new_fish_dict, new_board, score+fish))
        n += 1
    
    if len(scores)==0:
        return score
    else:
        return max(scores)
    
print(dfs(fish_dict, board, score))