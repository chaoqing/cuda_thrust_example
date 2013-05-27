#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

using namespace std;
using namespace boost::numeric::odeint;

/****************  使用的数据结构定义  ********************/
typedef float ElemType;//数据类型

typedef thrust::device_vector<ElemType> StateVector;//系统状态
typedef StateVector::iterator StateVectorIterator;//迭代器
typedef thrust::host_vector<ElemType> HStateVector;
typedef HStateVector::iterator HStateVectorIterator;

typedef StateVector ElemTypeVector;

typedef unsigned int Index;
typedef thrust::device_vector<Index> IndexVector;

/****************  一些系统常量定义  ********************/
const Index DIM=2;//系统维数
const Index N0=101;//系统大小
const Index REP=201;//系综大小
const Index N=N0*REP;//系统总大小
const Index LENGTH=N*DIM;//状态总长
const ElemType PI = 3.1415926535897932384626433832795029;
/****************  方程参数常量  ********************/
const ElemType mu=1.00;//分岔参数
const ElemType dr=0.15;//耦合参数实部
const ElemType di=-1.0;//耦合参数虚部
const ElemType b=1.0;//阻尼项
/****************  方程参数区间常量  ********************/
const ElemType omega_from=0.95*2*PI;
const ElemType omega_to=1.05*2*PI;
const ElemType couple_k_from=0.0;
const ElemType couple_k_to=10.0;
/****************  系统运行参数  ********************/
const ElemType dt = 1.0/360.0/1.0;//时间步长
const ElemType t_transients = 100.0;//暂态时长
const ElemType t_max = 300.0;//总时长

/****************  需要用到的(仿)函数  ********************/
struct plus_r//计算序参量
{
    template< class Tuple >
        __host__ __device__
        void operator()( Tuple t )
        {
            thrust::get<0>(t) += sqrt(thrust::get<1>(t)*thrust::get<1>(t)+thrust::get<2>(t)*thrust::get<2>(t));
        }
};

IndexVector rep_each(Index n, Index each)//[0,0,0, 1,1,1, 2,2,2, 3,3,3 ...] each=3
{
    IndexVector x(n);
    thrust::counting_iterator<Index> first(0);//计数迭代器
    thrust::transform(first, first+n,
            thrust::make_constant_iterator(each),//常量
            x.begin(),
            thrust::divides<Index>());
    return x;
}

IndexVector rep_times(Index n, Index len)//[0,1,2,...len-1, 0,1,2,...len-1, 0,1,2,...len-1]
{
    IndexVector x(n);
    thrust::counting_iterator<Index> first(0);//计数迭代器
    thrust::transform(first, first+n,
            thrust::make_constant_iterator(len),//常量
            x.begin(),
            thrust::modulus<Index>());
    return x;
}


void mean_field(//平均场
        ElemTypeVector &xmean, ElemTypeVector &ymean,//存放结果
        const StateVector &x, const IndexVector &group)//系统状态及系综分组
{
    static IndexVector useless(REP);//寄存系综序号集合,这个保存下来也没意思
    thrust::reduce_by_key(
            group.begin() ,group.end(),
            x.begin(), useless.begin(), xmean.begin(),
            thrust::equal_to<Index>(),
            thrust::plus<ElemType>());
    thrust::reduce_by_key(
            group.begin(),group.end(),
            x.begin()+N, useless.begin(), ymean.begin(),
            thrust::equal_to<Index>(),
            thrust::plus<ElemType>());

    thrust::transform(xmean.begin(), xmean.end(),
            thrust::make_constant_iterator((ElemType)N0),//常量
            xmean.begin(),
            thrust::divides<ElemType>());
    thrust::transform(ymean.begin(), ymean.end(),
            thrust::make_constant_iterator((ElemType)N0),//常量
            ymean.begin(),
            thrust::divides<ElemType>());
}

struct SYSTEM
{
    /****************  参数  ********************/
    ElemTypeVector k;//耦合强度
    ElemTypeVector omega;//自身频率
    ElemType o;//驱动频率
    ElemType f;//驱动强度

    /****************  一些需要用到的临时变量  ********************/
    ElemTypeVector xmean;//x坐标平均值
    ElemTypeVector ymean;//y坐标平均值
    IndexVector by_N0;
    IndexVector by_REP;

    /****************  设置参数  ********************/
    void set_omega(ElemType from=0.95*2*PI, ElemType to=1.05*2*PI){
        thrust::sequence(omega.begin(), omega.end(), from, (to-from)/(N0-1));
    }

    void set_k(ElemType from=0.0, ElemType to=1.0){
        thrust::sequence(k.begin(), k.end(), from, (to-from)/(REP-1));
    }


    SYSTEM(){
        omega.resize(N0);//各个子系统对应位置个体omega有相同值
        k.resize(REP);//各个子系统内部使用统一的耦合强度

        xmean.resize(REP);//记录各个子系统平均场
        ymean.resize(REP);
        by_N0=rep_times(N, N0);//按N0划分[0,1,2...N0-1,0,1,2...N0-1, ...... 0,1,2...N0-1] times=REP
        by_REP=rep_each(N, N0);//按系综划分[0,0,0,... 1,1,1,... 2,2,2..., ......,  REP-1,REP-1,REP-1,...] REP=N/N0,each=N0
    }

    struct function//单个振子的方程
    {
        ElemType dt;
        function(ElemType _dt):dt(_dt){}
        template< class Tuple >
            __host__ __device__
            void operator()( Tuple t )
            {
                /****************  定义一些宏来简化变量提取的过程  ********************/
#define value(x,index) thrust::get<index>((x))

#define x value(value(t,0),0)
#define y value(value(t,0),1)

#define xmean value(value(t,1),0)
#define ymean value(value(t,1),1)
#define k value(value(t,1),2)
#define omega value(value(t,1),3)
#define o value(value(t,1),4)
#define f value(value(t,1),5)

#define dxdt value(value(t,2),0)
#define dydt value(value(t,2),1)

                //ElemType x=value(value(t,0),0);
                //ElemType y=value(value(t,0),1);
                ElemType mo=x*x+y*y;

                dxdt = mu*x-omega*y + k*(dr*(xmean-x)-di*(ymean-y)) - b*mo*x+ f*cos(o*dt);
                dydt = mu*y+omega*x + k*(dr*(ymean-y)+di*(xmean-x)) - b*mo*y+ f*sin(o*dt);
#undef x
#undef y

#undef xmean
#undef ymean
#undef k
#undef omega
#undef o
#undef f

#undef dxdt
#undef dydt

#undef value
            }
    };

    void operator() ( const StateVector &x , StateVector &dxdt , const ElemType dt )
    {
        /****************  首先定义几个常用的操作  ********************/
#define ppi(x,y) /*permutation_parameter_iterator*/\
        thrust::make_permutation_iterator((x),(y)) //创建一个排列迭代器
#define Nspi(x) /*N_scale_parameter_iterator*/\
        thrust::make_constant_iterator((x)) //创建一个常量迭代器，在所有系综中通用参数
#define szi(x) /*state_zip_iterator*/\
        thrust::make_zip_iterator(thrust::make_tuple((x),(x)+N))//将各维分量拆开后打包
#define pzi(k,o,f,omega,xm,ym) /*parameter_zip_iterator*/\
        thrust::make_zip_iterator(thrust::make_tuple((k),(o),(f),(omega),(xm),(ym)))//将参数打包

        mean_field(xmean, ymean, x, by_REP);//更新平均场
        thrust::for_each(
                thrust::make_zip_iterator(//begin
                    thrust::make_tuple(
                        szi(x.begin()),//x
                        pzi(ppi(xmean.begin(),by_REP.begin()),//xmean
                            ppi(ymean.begin(),by_REP.begin()),//ymean
                            ppi(k.begin(), by_REP.begin()),//k
                            ppi(omega.begin(),by_N0.begin()),//omega
                            Nspi(o),//force omega
                            Nspi(f)//force strength
                           ),
                        szi(dxdt.begin()))),//dxdt 

                thrust::make_zip_iterator(//end
                    thrust::make_tuple(
                        szi(x.begin()+N),//x
                        pzi(ppi(xmean.begin(),by_REP.end()),//xmean
                            ppi(ymean.begin(),by_REP.end()),//ymean
                            ppi(k.begin(), by_REP.end()),//k
                            ppi(omega.begin(),by_N0.end()),//omega
                            Nspi(o),//force omega
                            Nspi(f)//force strength
                           ),
                        szi(dxdt.begin()+N))),//dxdt 

                function(dt)
                    );
#undef ppi
#undef Nspi
#undef szi
#undef pzi
    }
};


struct observer
{
    ElemTypeVector m_r;
    Index m_count;
    /****************  一些需要用到的临时变量  ********************/
    ElemTypeVector xmean;//x坐标平均值
    ElemTypeVector ymean;//y坐标平均值
    IndexVector by_N0;
    IndexVector by_REP;

    observer():m_count(0) {
        m_r.resize(REP);
        thrust::fill(m_r.begin(),m_r.end(),(ElemType)0.0);
        xmean.resize(REP);//记录各个子系统平均场
        ymean.resize(REP);
        by_N0=rep_times(N, N0);//按N0划分[0,1,2...N0-1,0,1,2...N0-1, ...... 0,1,2...N0-1] times=REP
        by_REP=rep_each(N, N0);//按系综划分[0,0,0,... 1,1,1,... 2,2,2..., ......,  REP-1,REP-1,REP-1,...] REP=N/N0,each=N0
    }

    template< class State >
        void operator()( const State &x , ElemType t )
        {
            mean_field(xmean, ymean, x, by_REP);//更新平均场
            thrust::for_each(
                    thrust::make_zip_iterator(//begin
                        thrust::make_tuple(
                            m_r.begin(),
                            xmean.begin(),
                            ymean.begin())), 

                    thrust::make_zip_iterator(//end
                        thrust::make_tuple(
                            m_r.end(),
                            xmean.end(),
                            ymean.end())), 
                    plus_r());
            ++m_count;
        }

    void report() {
        if(m_count!=0){
            thrust::transform(m_r.begin(), m_r.end(),
                    thrust::make_constant_iterator((ElemType)m_count),//常量
                    m_r.begin(),
                    thrust::divides<ElemType>());
        }
        thrust::copy(m_r.begin(),m_r.end(),std::ostream_iterator<ElemType>(std::cout,"\n"));
    }

    void reset( void ) {
        thrust::fill(m_r.begin(),m_r.end(),(ElemType)0.0);
        m_count = 0; 
    }
};


/****************  主函数开始  ********************/
int main( int arc , char* argv[] )
{
    HStateVector xinit_h( LENGTH );
    for( Index i=0 ; i<LENGTH ; ++i )
    {
        if(i<LENGTH/2&&i%N0==0)//x且一个新的子系统
        {
            xinit_h[i] = 0.1;
        }
        else{
            xinit_h[i] = 0.0;
        }
        //xinit_h[i] = drand48();
    }


    SYSTEM sys;
    sys.f=0.0;
    sys.o=0.0;

    typedef runge_kutta4< StateVector , ElemType , StateVector , ElemType > stepper_type;

    sys.set_k(couple_k_from,couple_k_to);
    sys.set_omega(omega_from,omega_to);
    
    observer obs;

    StateVector xinit = xinit_h;

    Index steps1 = integrate_const( stepper_type() , boost::ref( sys ) , xinit , (ElemType)0.0 , t_transients , dt );
    Index steps2 = integrate_const( stepper_type() , boost::ref( sys ) , xinit , (ElemType)0.0 , t_max , dt , boost::ref( obs ) );
    obs.report();

    return 0;
}
