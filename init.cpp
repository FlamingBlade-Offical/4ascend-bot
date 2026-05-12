/*
「東の空から始まる世界」
「始于东方之空的世界」

まだ見ぬ明日を見ようとしてた
我尝试着去摸索看不见的明天
君の声だけを聞いて
单单是听到你的声音
世界はこんなに広いのだと
我就恍然明白这个世界
気づかせてくれた
是这样的宽广
息を吸って見上げてみよう
轻轻吸一口气 抬头向上看的话
そこは綺麗な夢に見た場所
那里正是梦中看到的那美丽的地方
柔らかな風 吹いて
轻柔的风拂过身边……
幾億もの涙が作る世界
「无数的泪珠 方才组成了现在的世界」
ねぇ 気ついてたの
呐 你意识到这一点了吗
最初の言葉 繰り返して
当初的那两句话 反复在我们两人之间传递
いつでも笑ってたいよ
一直都这样笑着
小さいな迷いは空に消えた
小小的迷惑在空中消失了

触れ合いた指いつまででも
相互交合的手指一直到永远
伝わるようにと そうっと
悄悄的向你传达这份感情
二人確かにここにいると
两个人确实在这里留下了
印を残して
深深的印记
語り合ってた星空の日に
我们一起畅谈星空的那晚
約束をした 離れないだと
约定好了，彼此都不要离开
君は笑って泣いた
你一边笑一边哭着
見つけたいよ
想要再见到你
何度も願いを込めて
一直都这样祈愿着，这份心情超越了一切
大切にしてた心が今求めた世界
这份最重要的心情全世界都已寻求不到
愛しい気持ちだけは
我对你无限的爱意
もう届いてるの 君のもとへ
已经传递到你的心里去了吗？
優しい瞳に 見つめられたら
正因为你温柔的双眼注视着我
もどかしくなる ふたりの距離を
所以两人的距离逐渐变得躁动起来
囁きながら 近づいていくの
但却又在互相的耳语之中变得越来越近
願いはひとつ
而我们的愿望只有一个……
幾億もの涙が溢れたしてく
「无数的泪珠 从眼中夺眶而出」
そんな毎日を繰り返すように
就像是要重复这样的每个日子一样
僕らはまた 新しい夢を開く
我们再一次用自己的双手完成新的梦想
ねえ 二人で行こう
呐 我们两人一起上路吧
幾億もの涙が作る世界
「无数的泪珠 方才组成了现在的世界」
ねぇ 気ついてたの
呐 你意识到这一点了吗
最初の言葉 繰り返して
当初的那两句话 反复在我们两人之间传递
いつでも笑ってたいよ
无论何时 我都想将笑容奉献给你
どんな時だて傍にいるから
无论何时 我都会在你的身边
*/
#include <bits/stdc++.h>//喵内～
#include <random>
#define re register//喵内～
#define rep(i,a,b) for (re int i = (a);i <= (b); ++i)
#define debug(x) cout << #x << '=',print(x),putchar(' ')
#define file(x) freopen(x".in","r",stdin),freopen(x".out","w",stdout)
#define pi pair<int,int>
#define mp(a,b) make_pair(a,b)
#define SZ(x) ((int)(x).size())
#define all(x) (x).begin(),(x).end()
typedef long long ll;
using namespace std;//喵内～
inline ll read(){
    ll s = 0,f = 1;char c = getchar();
    while (!isdigit(c)){if (c == '-')f = -1;c = getchar();}
    while (isdigit(c)){s = (s<<3) + (s<<1) + (c ^ 48);c = getchar();}
    return s * f;
}//喵内～
void print(__int128 x){if (x < 0) {putchar('-'),print(-x);return ;}if (x >= 10) print(x / 10);putchar(x % 10 + 48);}//喵内～
const int Mod = 1e9 + 7;//喵内～要填数字哟～
//const int Mod = 998244353;//喵内～要填数字哟～
const ll INF = 0x3f3f3f3f;
const int N = 9;//喵内～要填数字哟～
ll qpow(ll x,ll y){
   ll res = 1;
    for (;y;y >>= 1,x = x * x % Mod) if (y & 1) res = res * x % Mod;
    return res;
}
//ATTENTION IS ALL YOU NEED
//DON'T GET STUCK ON ONE APPROACH!
std::random_device rd;
std::mt19937 rng(rd());
uniform_int_distribution<int> dist;
struct GameState{
    //基本内容
    int board[N][N]; // 存储棋盘状态，0 表示空白，1 表示白子，2 表示 黑子。
    int plants[N][N]; // 存储植物状态，0 表示无植物生长，1 表示有植物生长。
    int hp[3],cnt[3]; // 黑白双方血量，棋子个数
    int player_turn; // 轮到谁下

    //ascend 内容
    int ascend_turn; //当前谁先发动 ascend
    struct ascend{
        int attack; //攻击数
        vector<pair<int,int> > slots;//ascend 的棋子
        void init(){
            attack = 0;
            slots.clear();
        }
    }ascend_player[3];

    //植物生长内容
    int plant_top,plant_down,plant_left,plant_right;//植物生长范围
    int plant_range;//每次扩大范围
    double soil_fertility,max_fertility,growth_fertility; // 土壤肥力，肥力上限，每次生长消耗
    double fertility_recharge_rate;// 每次非 ascend 回合后土壤增加肥力。
    // Ascend 回合不会增加肥力，但在 ascend 结算后必定触发一次生长。

    void print_board(){
        cout << ascend_player[1].attack << " " << ascend_player[2].attack << endl;
        cout << hp[1] << " " << hp[2] << endl;
        for (int i = 0;i < N;++i){
            for (int j = 0;j < N;++j){
                cout << board[i][j] << " ";
            } 
            for (int j = 0;j < N;++j){
                if (plants[i][j]) cout << "*" << " ";
                else cout << "." << " ";
            }
            cout << endl;
        }
    }

    void init(){
        hp[1] = hp[2] = 6;
        player_turn = 1;//白
        cnt[1] = cnt[2] = 0;

        ascend_turn = 0;
        ascend_player[1].init();
        ascend_player[2].init();

        plant_top = plant_down = plant_left = plant_right = 4;
        plant_range = 2;
        soil_fertility = 0; max_fertility = 10; growth_fertility = 1.4;
        fertility_recharge_rate = 0.83;
    };

    int coords_check(int x,int y){
        if (x < 0 || y < 0 || x >= N || y >= N ) return 0;
        return 1;
    }

    int dx[4] = {0,1,1,1};
    int dy[4] = {1,1,0,-1};
    int ascend_check(int x,int y,int op){
        for (int i = 0;i < 4;++i){
            int positive_cnt = 0,negative_cnt = 0;
            while (coords_check(x + positive_cnt * dx[i],y + positive_cnt * dy[i])
                && board[x + positive_cnt * dx[i]][y + positive_cnt * dy[i]] == op)
                ++positive_cnt;
            while (coords_check(x - negative_cnt * dx[i],y - negative_cnt * dy[i])
                && board[x - negative_cnt * dx[i]][y - negative_cnt * dy[i]] == op)
                ++negative_cnt;
            
            //判断连成的正长和负长
            if (positive_cnt + negative_cnt >= 5){
                for (int j = 1;j < positive_cnt;++j){
                    ascend_player[op].attack += 1 + plants[x + j * dx[i]][y + j * dy[i]];
                    ascend_player[op].slots.push_back(mp(x + j * dx[i],y + j * dy[i]));
                }
                for (int j = 1;j < negative_cnt;++j){
                    ascend_player[op].attack += 1 + plants[x - j * dx[i]][y - j * dy[i]];
                    ascend_player[op].slots.push_back(mp(x - j * dx[i],y - j * dy[i]));
                }
            }
        }
        if (ascend_player[op].attack != 0){
            ascend_player[op].attack += 1 + plants[x][y];
            ascend_player[op].slots.push_back(mp(x,y));
            for (auto &[x,y] : ascend_player[op].slots) board[x][y] = 0;
            return 1;
        } return 0;
    }

    int game_end_check(){
        if (hp[1] < 0){cout << "Black wins because white hp is below zero!" << endl; return 1;}
        if (hp[2] < 0){cout << "White wins because black hp is below zero!" << endl; return 1;}
        if (cnt[1] + cnt[2] == N * N){
            if (hp[1] < hp[2]) {
                cout << "Black wins because it has more hp than white!" << endl; 
                return 1;
            }
            if (hp[2] < hp[1]) {
                cout << "White wins because it has more hp than black!" << endl; return 1;
            }
            if (hp[1] == hp[2]){
                if (cnt[1] < cnt[2]){
                    cout << "Black wins because it has more ramaining pieces!" << endl; 
                    return 1;
                }
                if (cnt[2] < cnt[1]){
                    cout << "White wins because it has more ramaining pieces!" << endl; 
                    return 1;
                }
            }
        } return 0;
    }
    
    void generate_plant(){
        plant_down = plant_right = 4;
        plant_left = plant_top = 4;
        for (int i = 0;i < N;++i)
            for (int j = 0;j < N;++j){
                if (board[i][j] || plants[i][j]){
                    plant_down = max(i,plant_down);
                    plant_top = min(i,plant_top);
                    plant_right = max(j,plant_right);
                    plant_left = min(j,plant_left);
                }
            }
        plant_down = min(N - 1,plant_down + plant_range);
        plant_right = min(N - 1,plant_right + plant_range);
        plant_left = max(0,plant_left - plant_range);
        plant_top = max(0,plant_top - plant_range);

        int max_grow_number = (int)(1.0 * soil_fertility / growth_fertility);//最大生成株数
        int min_grow_number = (int)(0.3 * soil_fertility / growth_fertility);//最小生成株数
        int grow_number = min_grow_number + dist(rng) % (max_grow_number - min_grow_number + 1);//最终生成株数

        vector<pair<int,int> > blank_space;
        for (int i = plant_top; i <= plant_down; ++i)
            for (int j = plant_left; j <= plant_right; ++j){
                if (board[i][j] == 0 && plants[i][j] == 0) blank_space.push_back(mp(i,j));
            }

        int siz = blank_space.size() - 1;
        grow_number = min(grow_number,siz);
        for (int i = 0;i < grow_number;++i){
            //i ~ siz 随机选一个数
            int j = i + dist(rng) % (siz - i + 1);
            plants[blank_space[j].first][blank_space[j].second] = 1;
            swap(blank_space[i],blank_space[j]);
        } 
        soil_fertility -= 1.0 * grow_number * growth_fertility;
        return ;
    }

    void game(){
        init();
        while (1){
            //print_board();
            int px = 0,py = 0;
            cin >> px >> py;
            if (!coords_check(px,py) || board[px][py]){// 输入位置不合法
                cout << "input is invalid" << endl;
                return ;
            }//成功读入坐标

            board[px][py] = player_turn; cnt[player_turn]++;

            if (ascend_check(px,py,player_turn)){
                if (ascend_turn == 0){
                    ascend_turn = player_turn;
                    player_turn = 3 - player_turn;
                    continue;
                }
            }
            if (ascend_turn){//ascend 回合结算
                int decrease = 0;// 计算先手被防御了多少
                for (auto &[x,y] : ascend_player[ascend_turn].slots){
                    if (x == px && y == py){
                        if (plants[px][py] == 1) decrease = 2;
                        else decrease = 1;
                        break;
                    }
                }
                ascend_player[ascend_turn].attack -= decrease;
                //进行伤害结算
                hp[1] -= ascend_player[2].attack - min(ascend_player[1].attack,ascend_player[2].attack);
                hp[2] -= ascend_player[1].attack - min(ascend_player[1].attack,ascend_player[2].attack);

                // 移除所有相关格子和植物
                for (int i = 1;i <= 2;++i)
                    for (auto &[x,y] : ascend_player[i].slots) plants[x][y] = 0;
                cnt[1] -= ascend_player[1].slots.size();
                cnt[2] -= ascend_player[2].slots.size();

                ascend_player[1].init();
                ascend_player[2].init();
            }
            if (game_end_check()) break;//检查是否有人没血或者下完了
            //植物生成
            if (ascend_turn){
                generate_plant();
                ascend_turn = 0;
            }
            else {
                soil_fertility += fertility_recharge_rate;
                soil_fertility = min(soil_fertility,max_fertility);
                if (soil_fertility == max_fertility) generate_plant();
            }
            player_turn = 3 - player_turn;
        }
        return ;
    }
}B;
signed main(){
    B.game();
    return 0;
}//喵内～