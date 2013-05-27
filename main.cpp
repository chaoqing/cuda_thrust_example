#include <iostream>
#include <fstream>
#include <utility>

#include <boost/numeric/odeint.hpp>
using namespace std;
using namespace boost::numeric::odeint;

/****************  使用的数据结构定义  ********************/
typedef float ElemType;//数据类型
typedef unsigned int Index;
typedef vector< ElemType > StateVector;
typedef StateVector::iterator StateVectorIterator;//迭代器
typedef StateVector ElemTypeVector;
/****************  一些常量定义  ********************/
const Index DIM=2;//系统维数
const Index N=1000;//系统大小
const Index LENGTH=N*DIM;//状态总长
const ElemType PI = 3.1415926535897932384626433832795029;


/****************  方程参数常量  ********************/
const ElemType mu=1.00;//分岔参数
const ElemType dr=0.15;//耦合参数实部
const ElemType di=-1.0;//耦合参数虚部
const ElemType b=1.0;//阻尼项

pair< ElemType , ElemType > mean_field( const StateVector &x )
{
    ElemType xmean=0,ymean=0;
    for( Index i=0 ; i<N ; ++i )
    {
        xmean+=x[i];
        ymean+=x[i+N];
    }

    return make_pair( xmean/(ElemType)N , ymean/(ElemType)N );
}


struct SYSTEM
{
    /****************  参数  ********************/
    ElemType k;//耦合强度
    ElemTypeVector omega;//自身频率
    ElemType o;//驱动频率
    ElemType f;//驱动强度

    /****************  设置参数  ********************/
    void set_omega(ElemType from=0.95*2*PI, ElemType to=1.05*2*PI){
        for (Index i = 0; i < N; i++) {
            omega[i]=from+i*(to-from)/(N-1);
        }
    }


    SYSTEM(){
        omega.resize(N);//各个子系统对应位置个体omega有相同值
    }

    void operator() ( const StateVector &x , StateVector &dxdt , const ElemType dt )
    {
        pair<ElemType,ElemType> meanf=mean_field(x);//更新平均场
        for (Index i = 0; i < N; i++) {
                ElemType mo=x[i]*x[i]+x[i+N]*x[i+N];

                dxdt[i] = mu*x[i]-omega[i]*x[i+N]
                    + k*(dr*(meanf.first-x[i])-di*(meanf.second-x[i+N]))
                    - b*mo*x[i]/* + f*cos(o*dt)*/;
                dxdt[i+N] = mu*x[i+N]+omega[i]*x[i]
                    + k*(dr*(meanf.second-x[i+N])+di*(meanf.first-x[i]))
                    - b*mo*x[i+N]/*+ f*sin(o*dt)*/;
        }
    }
};

struct observer
{
    ElemType m_r;
    Index m_count;

    observer():m_r(0.0),m_count(0) {
    }

    template< class State >
        void operator()( const State &x , ElemType t )
        {
            pair<ElemType,ElemType> meanf=mean_field(x);//更新平均场
            m_r+=sqrt(meanf.first*meanf.first+meanf.second*meanf.second);
            ++m_count;
        }

    void report() {
        if(m_count!=0){
            m_r/=(ElemType)m_count;
        }
        cout<<m_r<<endl;
    }
    void reset(){
        m_r=0.0;
        m_count=0;
    }
};

/****************  系统运行参数  ********************/
const ElemType dt = 0.01;//时间步长
const ElemType t_transients = 5.0;//暂态时长
const ElemType t_max = 40.0;//总时长

int main( int argc , char **argv )
{
    SYSTEM sys;

    sys.set_omega();
    sys.k=0.0;
    sys.f=0;
    sys.o=0;
    
    observer obs;
    StateVector x_h( LENGTH );

    for (sys.k = 0.1; sys.k<3.0001; sys.k+=0.10) {
        for( Index i=0 ; i<LENGTH ; ++i )
        {
            if(i<LENGTH/2)//x
            {
                x_h[i] = 1.0;
            }
            else{//y
                x_h[i] = 0.0;
            }
            //x_h[i] = drand48();
        }
        x_h[0] = 0.0;//让子系统第一个点的x坐标有个偏差

        //Index steps1 = integrate_const( runge_kutta4< StateVector >() , boost::ref( sys ) , x_h , (ElemType)0.0 , t_transients , dt );
        Index steps2 = integrate_const( runge_kutta4< StateVector >() , boost::ref( sys ) , x_h , (ElemType)0.0 , t_max , dt, boost::ref( obs ) );
        cout<< sys.k<<"\t";
        obs.report();
        obs.reset();
    }


    return 0;
}
