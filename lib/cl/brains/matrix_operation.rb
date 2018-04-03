
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

            def introspect
                { cl_inputs: [ :cl_matrix_a, :cl_matrix_b ], inputs: [:matrix_a, :matrix_b], name: self.class.name }
            end

            def execute
                @matrix_c = NArray.sfloat(@cols * @rows)
                @b_matrix_c = @context.create_buffer(@matrix_c.size * @matrix_c.element_size)

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
                native_matrix_a = NArray.sfloat(matrix_a.size * matrix_a[0].size)
                native_matrix_b = NArray.sfloat(matrix_b.size * matrix_b[0].size)

                @cols = matrix_b[0].size
                @rows = matrix_a.size
                @width = matrix_a[0].size

                matrix_a.flatten.each_with_index do |element, index|
                    native_matrix_a[index] = element
                end
          
                matrix_b.flatten.each_with_index do |element, index|
                    native_matrix_b[index] = element
                end

                @cl_matrix_a = @context.create_buffer(native_matrix_a.size * native_matrix_a.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => native_matrix_a)
                @cl_matrix_b = @context.create_buffer(native_matrix_b.size * native_matrix_b.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => native_matrix_b)
            end

            def rubynize
                @queue.enqueue_read_buffer(@b_matrix_c, @matrix_c, :event_wait_list => @events)
                @queue.finish
                @matrix_c.to_a.each_slice(@cols).collect { |slice| slice }
            end
        end
    end
end