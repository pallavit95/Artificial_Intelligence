load both the projects individually in visual studio code or any other ide

For BFS, in the search folder run the following in the command line
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
python pacman.py -l bigMaze2 -p SearchAgent -a fn=bfs -z .5
python pacman.py -l bigMaze3 -p SearchAgent -a fn=bfs -z .5


For DFS, in the search folder run the following in the command line
python pacman.py -l bigMaze -p SearchAgent -a fn=dfs -z .5
python pacman.py -l bigMaze2 -p SearchAgent -a fn=dfs -z .5
python pacman.py -l bigMaze3 -p SearchAgent -a fn=dfs -z .5

For A*, in the search folder run the following in the command line
python pacman.py -l bigMaze -z 0.5  -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l bigMaze2 -z 0.5  -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l bigMaze3 -z 0.5  -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

python pacman.py -l bigMaze -z 0.5  -p SearchAgent -a fn=astar,heuristic=euclideanHeuristic
python pacman.py -l bigMaze2 -z 0.5  -p SearchAgent -a fn=astar,heuristic=euclideanHeuristic
python pacman.py -l bigMaze3 -z 0.5  -p SearchAgent -a fn=astar,heuristic=euclideanHeuristic

For Policy Iteration, in the reinforcement folder, run the following in the command line
python gridworld.py -a policy -i 55 -k 10 -g MazeGrid -w 100
python gridworld.py -a policy -i 55 -k 10 -g MazeGrid2 -w 100
python gridworld.py -a policy -i 55 -k 10 -g MazeGrid3 -w 100

For Value Iteration, in the reinforcement folder, run the following in the command line
python gridworld.py -a value -i 55 -k 10 -g MazeGrid -w 100
python gridworld.py -a value -i 55 -k 10 -g MazeGrid2 -w 100
python gridworld.py -a value -i 55 -k 10 -g MazeGrid3 -w 100