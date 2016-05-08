#include <iostream>
#include <algorithm>
#include <condition_variable>
#include <thread>
#include <sstream>

#include <vtkVersion.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkChartXY.h>
#include <vtkTable.h>
#include <vtkPlot.h>
#include <vtkFloatArray.h>
#include <vtkContextView.h>
#include <vtkContextScene.h>
#include <vtkPen.h>
#include <vtkCommand.h>
#include <vtkAxis.h>

#include "BurnModelSolver/BurnModelSolver.hpp"


bool stop_work = false;
bool done_work = false;
bool update_y = false;

std::condition_variable check_update;
std::mutex lock_y;

std::vector<float> y;

void work(Solver& solver)
{
    std::cout << "Worker start" << std::endl;
    try {
        while (!stop_work) {
            if (!done_work) {
                if (!solver.is_done()) {
                    solver.step();
                }
                else {
                    done_work = true;
                }
            }

            if (update_y) {
                std::unique_lock<std::mutex> l(lock_y);
                y = solver.get_y();
                update_y = false;
                check_update.notify_one();
            }
        }
    }
    catch (std::exception& e) {
        std::cerr << "Exception caught in worker: " << e.what() << std::endl;
        done_work = true;
    }

    std::cout << "Worker done, wait for join" << std::endl;
}

class TimerCallback: public vtkCommand {
    public:
        static TimerCallback* New()
        {
            TimerCallback* cb = new TimerCallback;
            return cb;
        }

        virtual void Execute(vtkObject* caller, unsigned long eventId, void* callData)
        {
            std::cout << "Timer rerender" << std::endl;

            bool last_callback = false;
            auto iren = vtkRenderWindowInteractor::SafeDownCast(caller);

            if (done_work) {
                iren->RemoveObserver(observer_tag);
                last_callback = true;
            }

            std::unique_lock<std::mutex> l(lock_y);
            update_y = true;

            //wait notification or 100ms while update_y == true
            if (check_update.wait_for(l, std::chrono::milliseconds(100), [] {return !update_y;})) {
                if (arr_y) {
                    arr_y->SetArray(y.data(), (vtkIdType) y.size(), 1);
                }
                if (table) {
                    table->Modified();
                }

                iren->GetRenderWindow()->Render();
            }

            if (last_callback) {
                stop_work = true;
                std::cout << "Timer rerender done" << std::endl;
            }
        }

        void set_array(vtkFloatArray* new_array)
        {
            arr_y = new_array;
        }

        void set_table(vtkTable* new_table)
        {
            table = new_table;
        }

        void set_observer_tag(unsigned long new_tag)
        {
            observer_tag = new_tag;
        }

    private:
        vtkSmartPointer<vtkFloatArray> arr_y = nullptr;
        vtkSmartPointer<vtkTable> table = nullptr;
        unsigned long observer_tag = 0;
};

int main()
{
    size_t N = 4096;
    float x1 = 1e-2f;
    float tau = 1e-8f;
    float t_end = 1e0f;


    Params params;
    params.rho_t = 1600.f;
    params.C_pt = 1464.f;
    params.T_s = 720.f;
    params.G_t = 0.f;

    params.T_max = 2372.f;
    params.R_max = 363.f;
    params.C_p = 1800.f;
    params.lambda = 0.23f;

    params.A_k = 1e9f;
    params.E_a = 4.2e6f;

    params.R_s = 250.f;
    params.T_s0 = 300.f;

    params.q_r = 0.f;
    params.p_k = 1e6f;
    params.create_params();


    BurnModelSolver solver(N);
    solver.set_params(params);
    solver.set_x1(x1);
    solver.set_tau(tau);
    solver.set_t_end(t_end);

    std::vector<float> x = solver.get_x();

    y.resize(N);


    auto worker_thread = std::thread(work, std::ref(solver));


    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> arr_x = vtkSmartPointer<vtkFloatArray>::New();
    arr_x->SetName("X");
    table->AddColumn(arr_x);

    vtkSmartPointer<vtkFloatArray> arr_y = vtkSmartPointer<vtkFloatArray>::New();
    arr_y->SetName("Y");
    table->AddColumn(arr_y);

    arr_x->SetArray(x.data(), (vtkIdType) x.size(), 1);
    arr_y->SetArray(y.data(), (vtkIdType) y.size(), 1);


    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    chart->GetAxis(vtkAxis::LEFT)->SetRange(0.f, 1.f);
    chart->GetAxis(vtkAxis::LEFT)->SetBehavior(vtkAxis::FIXED);
    chart->GetAxis(vtkAxis::BOTTOM)->SetRange(0.f, x1);
    chart->GetAxis(vtkAxis::BOTTOM)->SetBehavior(vtkAxis::FIXED);

    view->GetScene()->AddItem(chart);


    vtkPlot* line = chart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, 1);

    line->SetColor(0, 255, 0, 255);
    line->SetWidth(1.0);
    line = chart->AddPlot(vtkChart::LINE);
    line->GetPen()->SetLineType(vtkPen::DASH_LINE);


    auto iren = view->GetInteractor();
    iren->Initialize();

    auto callback = vtkSmartPointer<TimerCallback>::New();
    callback->set_array(arr_y);
    callback->set_table(table);

    auto observer_tag = iren->AddObserver(vtkCommand::TimerEvent, callback);
    callback->set_observer_tag(observer_tag);

    iren->CreateRepeatingTimer(1000);

    iren->Start();


    stop_work = true;
    worker_thread.join();

    std::stringstream path("/home/tsv/Документы/burn_model");

    using std::chrono::system_clock;
    std::time_t tt = system_clock::to_time_t(system_clock::now());
    std::tm* ptm = std::localtime(&tt);

    std::stringstream time;
    time << std::put_time(ptm, "%F %H-%M");

    solver.save(path.str(), time.str());
    std::cout << "Save done" << std::endl;

    std::cout << "Join done, quit" << std::endl;
    return 0;
}