class amgxFields { 

  public: 
    AMGX_Mode mode = AMGX_mode_dDDI;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;

    double *result;
    bool initDone;

    amgxFields():
        result(NULL)
    {
        initDone = false;
    }

    ~amgxFields()
    {
        reset();
    }

    void init()
    {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, "./system/config"));
        AMGX_resources_create_simple(&rsrc, cfg);
        AMGX_matrix_create(&A, rsrc, mode);
        AMGX_vector_create(&x, rsrc, mode);
        AMGX_vector_create(&b, rsrc, mode);
        AMGX_solver_create(&solver, rsrc, mode, cfg);
        initDone = true;
    }

    void setup( int* rPtr,
                int* col,
                double* val,
                double* source,
                int nCells,
                int nIFaces)
    {
        if (!initDone)
        {
             FatalErrorInFunction
                << "ERROR: setup is called before init "
                << endl
                << abort(FatalError);
        }
        
        int bsize_x = 1;
        int bsize_y = 1;
        AMGX_matrix_upload_all(A, nCells, nCells+2*(nIFaces), bsize_x, bsize_y, &rPtr[0], &col[0], &val[0], NULL);
        AMGX_solver_setup( solver, A );
        AMGX_vector_upload( b, nCells, bsize_y, &source[0] );
        AMGX_vector_set_zero(x, nCells, bsize_x);
    }

    void solve()
    {
        AMGX_solver_solve_with_0_initial_guess( solver, b, x );
    }

    void getSolution( int nCells, double*&h_T_new)
    {
        //cudaMalloc(reinterpret_cast<void**>(&result),(nCells*sizeof(double)));
        result = static_cast<double*>(malloc(nCells*sizeof(double)));
        AMGX_vector_download(x, result);
        h_T_new=result;
    }
    void reset()
    {
        AMGX_solver_destroy(solver);
        AMGX_vector_destroy(x);
        AMGX_vector_destroy(b);
        AMGX_matrix_destroy(A);
        AMGX_resources_destroy(rsrc);
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
        AMGX_SAFE_CALL(AMGX_finalize())
        
        if (result) free(result);
        initDone = false;
    }

}; 
