
module Cl
    module Brains
        class MatrixOperation
            attr_accessor :cl_matrix_a, :cl_matrix_b, :cols, :rows, :width

            def initialize(context, prog, queue, events = [])
                @context = context
                @queue = queue
                @prog = prog
                @events = events || []
            end

            def program!
                @@matrix_source ||= File.read(File.join(File.dirname(__FILE__), 'matrix.cl'))
                @prog = @context.create_program_with_source( @@matrix_source )
                @prog.build
            end

            def introspect
                { cl_inputs: [ :cl_matrix_a, :cl_matrix_b ], inputs: [:matrix_a, :matrix_b], name: self.class.name }
            end

            def execute
                @output_tensor = Cl::Brains::Tensor.new(:float, 2, {cols: @cols, rows: @rows} )
                @b_matrix_c = @output_tensor.open_cl_buffer(@context)

                m = OpenCL::Int1::new(@rows)
                n = OpenCL::Int1::new(@cols)
                k = OpenCL::Int1::new(@width)
                @events << @prog.matrix(@queue, [@rows, @cols], m, n, k, @cl_matrix_a, @cl_matrix_b, @b_matrix_c)
                self
            end

            def result
                { matrix: @b_matrix_c, row: @rows, cols: @cols }
            end

            # generate native buffers from params
            def nativize(matrix_a:, matrix_b:)
                @cols = matrix_b.shape.cols
                @rows = matrix_a.shape.rows
                @width = matrix_a.shape.cols

                @cl_matrix_a = matrix_a.open_cl_buffer(@context)
                @cl_matrix_b = matrix_b.open_cl_buffer(@context)
            end

            def rubynize
                @output_tensor.sync_cl_buffer(@queue, @events)
                @queue.finish
                @output_tensor
            end
        end
    end
end