from math import cos

import numpy as np
from numba import cuda

from pylops.utils.backend import to_cupy


class _KirchhoffCudaHelper:
    """A helper class for performing Kirchhoff migration or modeling using Numba CUDA.

    This class provides methods to compute the forward and adjoint operations for the
    Kirchhoff operator, utilizing GPU acceleration through Numba's CUDA capabilities.

    Parameters
    ----------
    ns : :obj:`int`
        Number of sources.
    nr : :obj:`int`
        Number of receivers.
    nt : :obj:`int`
        Number of time samples.
    ni : :obj:`int`
        Number of image points.
    dynamic : :obj:`bool`, optional
        Flag indicating whether to use dynamic computation (default is `False``).
    nstreams : :obj:`int`, optional
        Number of streams used in case of ``dynamic=True``.

    """

    def __init__(self, ns, nr, nt, ni, dynamic=False, nstreams=1):
        self.ns, self.nr, self.nt, self.ni = ns, nr, nt, ni
        self.dynamic = dynamic
        self.nstreams = nstreams

        self._grid_setup()

    def _grid_setup(self):
        """Set up the CUDA grid and block dimensions based on the current device and computation flags.

        This method configures the number of blocks and threads per block for CUDA kernels,
        depending on whether dynamic weights are ued in computations or not.
        """
        current_device = cuda.get_current_device()
        if self.dynamic:
            num_sources = self.ns
            num_streams = self.nstreams
            streams = [cuda.stream() for _ in range(num_streams)]
            sources_per_stream = num_sources // num_streams
            remainder = num_sources % num_streams
            source_counter = 0
            self.sources_per_streams = {}
            for i, stream in enumerate(streams):
                num_sources_for_stream = sources_per_stream + (
                    1 if i < remainder else 0
                )
                sources_for_stream = list(
                    range(source_counter, source_counter + num_sources_for_stream)
                )
                self.sources_per_streams[stream] = sources_for_stream
                source_counter += num_sources_for_stream
            n_runs = (
                num_streams * self.ns * self.ni * self.nr
            )  # number_of_times_to_run_kernel
            self.num_threads_per_blocks = current_device.WARP_SIZE * 8
            self.num_blocks = (
                n_runs + (self.num_threads_per_blocks - 1)
            ) // self.num_threads_per_blocks
        else:
            warp = current_device.WARP_SIZE
            self.num_threads_per_blocks = (warp, warp)
            self.num_blocks = (
                (self.ns + self.num_threads_per_blocks[0] - 1)
                // self.num_threads_per_blocks[0],
                (self.nr + self.num_threads_per_blocks[1] - 1)
                // self.num_threads_per_blocks[1],
            )

    def _data_prep_dynamic(
        self,
        ns,
        nr,
        nt,
        ni,
        nz,
        dt,
        aperture,
        angleaperture,
        aperturetap,
        vel,
        six,
        rix,
        trav_recs,
        angle_recs,
        trav_srcs,
        angle_srcs,
        amp_srcs,
        amp_recs,
    ):
        """Data preparation for dynamic computation

        Prepare data for dynamic computation by transfering some variables to device
        memory in advance once.
        """
        ns_d = np.int32(ns)
        nr_d = np.int32(nr)
        nt_d = np.int32(nt)
        ni_d = np.int32(ni)
        nz_d = np.int32(nz)
        dt_d = np.float32(dt)
        aperturemin_d = np.float32(aperture[0])
        aperturemax_d = np.float32(aperture[1])
        angleaperturemin_d = np.float32(angleaperture[0])
        angleaperturemax_d = np.float32(angleaperture[1])
        aperturetap_d = cuda.to_device(aperturetap)
        vel_d = cuda.to_device(vel)
        six_d = cuda.to_device(six)
        rix_d = cuda.to_device(rix)
        self.const_inputs = (
            ns_d,
            nr_d,
            nt_d,
            ni_d,
            nz_d,
            dt_d,
            aperturemin_d,
            aperturemax_d,
            angleaperturemin_d,
            angleaperturemax_d,
            vel_d,
            aperturetap_d,
            six_d,
            rix_d,
        )

        self.trav_recs_d_global = cuda.to_device(trav_recs)
        self.angles_recs_d_global = cuda.to_device(angle_recs)
        self.trav_srcs_d_global = cuda.to_device(trav_srcs)
        self.angles_srcs_d_global = cuda.to_device(angle_srcs)
        self.amp_srcs_d_global = cuda.to_device(amp_srcs)
        self.amp_recs_d_global = cuda.to_device(amp_recs)

    def _call_nondynamic(self, opt, *inputs):
        """Process data using CUDA single stream for non-dynamic computation.

        This method handles data preparation and execution of CUDA kernels
        for both forward and adjoint operations of non-dynamic case.

        Parameters
        ----------
        opt : :obj:`str`
            Operation type, either '_matvec' for forward or '_rmatvec' for adjoint.
        *inputs : :obj:`list`
            List of input parameters required by the kernels.

        Returns
        -------
        y_d : :obj:`numpy.ndarray`
            Output data after processing.

        """

        if opt == "_matvec":
            x_d = inputs[0]
            y_d = inputs[1]
            ns_d = np.int32(inputs[2])
            nr_d = np.int32(inputs[3])
            nt_d = np.int32(inputs[4])
            ni_d = np.int32(inputs[5])
            dt_d = np.float32(inputs[6])
            trav_srcs_d = to_cupy(inputs[7])
            trav_recs_d = to_cupy(inputs[8])
            self._travsrcrec_kirch_matvec_cuda[
                self.num_blocks, self.num_threads_per_blocks
            ](x_d, y_d, ns_d, nr_d, nt_d, ni_d, dt_d, trav_srcs_d, trav_recs_d)
            cuda.synchronize()

        elif opt == "_rmatvec":
            x_d = inputs[0]
            y_d = inputs[1]
            ns_d = np.int32(inputs[2])
            nr_d = np.int32(inputs[3])
            nt_d = np.int32(inputs[4])
            ni_d = np.int32(inputs[5])
            dt_d = np.float32(inputs[6])
            trav_srcs_d = to_cupy(inputs[7])
            trav_recs_d = to_cupy(inputs[8])
            self._travsrcrec_kirch_rmatvec_cuda[
                self.num_blocks, self.num_threads_per_blocks
            ](x_d, y_d, ns_d, nr_d, nt_d, ni_d, dt_d, trav_srcs_d, trav_recs_d)
            cuda.synchronize()

        return y_d

    def _call_dynamic(self, x, opt):
        """Process data using CUDA streams for dynamic computation.

        This method handles data preparation and execution of CUDA kernels using streams,
        for both forward and adjoint operations of dynamic case.

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Input data (image or seismic data).
        opt : :obj:`str`
            Operation type, either '_matvec' for forward or '_rmatvec' for adjoint.

        Returns
        -------
        y : :obj:`numpy.ndarray`
            Output data after processing.

        """

        if opt == "_matvec":
            x = x.ravel()
            y = np.zeros((self.ns * self.nr, self.nt))
        elif opt == "_rmatvec":
            y = np.zeros(self.ni)

        y_d_dict = {}
        isrc_list_d_dict = {}
        for stream, isrc_list in self.sources_per_streams.items():
            x_d = cuda.to_device(x, stream=stream)
            y_d_dict[stream] = cuda.to_device(y, stream=stream)
            isrc_list_d_dict[stream] = cuda.to_device(isrc_list, stream=stream)

        for stream in self.sources_per_streams.keys():
            stream.synchronize()
        for irec in range(self.nr):
            for stream, isrc_list in self.sources_per_streams.items():
                if opt == "_matvec":
                    self._ampsrcrec_kirch_matvec_cuda_streams[
                        self.num_blocks, self.num_threads_per_blocks, stream
                    ](
                        *self.const_inputs,
                        self.trav_srcs_d_global,
                        self.amp_srcs_d_global,
                        self.angles_srcs_d_global,
                        self.trav_recs_d_global,
                        self.amp_recs_d_global,
                        self.angles_recs_d_global,
                        y_d_dict[stream],
                        isrc_list_d_dict[stream],
                        irec,
                        x_d
                    )
                elif opt == "_rmatvec":
                    self._ampsrcrec_kirch_rmatvec_cuda_streams[
                        self.num_blocks, self.num_threads_per_blocks, stream
                    ](
                        *self.const_inputs,
                        self.trav_srcs_d_global,
                        self.amp_srcs_d_global,
                        self.angles_srcs_d_global,
                        self.trav_recs_d_global,
                        self.amp_recs_d_global,
                        self.angles_recs_d_global,
                        y_d_dict[stream],
                        isrc_list_d_dict[stream],
                        irec,
                        x_d
                    )
        # Synchronize the streams to ensure all operations have been completed
        for stream in self.sources_per_streams.keys():
            stream.synchronize()
        y_streams = []
        for stream, y_dev in y_d_dict.items():
            y_streams.append(y_dev.copy_to_host(stream=stream))
        y_total = np.sum(y_streams, axis=0)
        return y_total

    @staticmethod
    @cuda.jit
    def _travsrcrec_kirch_matvec_cuda(x, y, ns, nr, nt, ni, dt, trav_srcs, trav_recs):
        isrc, irec = cuda.grid(2)
        if ns > isrc and nr > irec:
            for ii in range(ni):
                travisrc = trav_srcs[ii, isrc]
                travirec = trav_recs[ii, irec]
                trav = travisrc + travirec
                itravii = int(trav / dt)
                travdii = trav / dt - itravii
                if 0 <= itravii < nt - 1:
                    ind1 = (isrc * nr + irec, itravii)
                    val1 = x[ii] * (1 - travdii)
                    ind2 = (isrc * nr + irec, itravii + 1)
                    val2 = x[ii] * travdii
                    cuda.atomic.add(y, ind1, val1)
                    cuda.atomic.add(y, ind2, val2)

    @staticmethod
    @cuda.jit
    def _travsrcrec_kirch_rmatvec_cuda(x, y, ns, nr, nt, ni, dt, trav_srcs, trav_recs):
        isrc, irec = cuda.grid(2)
        if ns > isrc and nr > irec:
            for ii in range(ni):
                trav_srcsii = trav_srcs[ii]
                trav_recsii = trav_recs[ii]
                trav_srcii = trav_srcsii[isrc]
                trav_recii = trav_recsii[irec]
                travii = trav_srcii + trav_recii
                itravii = int(travii / dt)
                travdii = travii / dt - itravii
                if 0 <= itravii < nt - 1:
                    vii = (
                        x[isrc * nr + irec, itravii] * (1 - travdii)
                        + x[isrc * nr + irec, itravii + 1] * travdii
                    )
                    cuda.atomic.add(y, ii, vii)

    @staticmethod
    @cuda.jit
    def _ampsrcrec_kirch_matvec_cuda_streams(
        ns,
        nr,
        nt,
        ni,
        nz,
        dt,
        aperturemin,
        aperturemax,
        angleaperturemin,
        angleaperturemax,
        vel,
        aperturetap,
        six_d,
        rix_d,
        travsrc,
        ampsrc,
        anglesrc,
        travi,
        ampi,
        anglei,
        y,
        isrc_list,
        irec,
        x,
    ):
        ii = cuda.grid(1)
        if ni > ii:
            index_isrc = -1
            for isrc in isrc_list:
                index_isrc = index_isrc + 1
                sixisrcrec = six_d[isrc * nr + irec]
                rixisrcrec = rix_d[isrc * nr + irec]
                travirec = travi[:, irec]
                ampirec = ampi[:, irec]
                angleirec = anglei[:, irec]
                travisrc = travsrc[:, isrc]
                ampisrc = ampsrc[:, isrc]
                angleisrc = anglesrc[:, isrc]
                daperture = aperturemax - aperturemin
                dangleaperture = angleaperturemax - angleaperturemin
                trav = travisrc[ii] + travirec[ii]
                itravii = int(trav / dt)
                travdii = trav / dt - itravii
                # compute cosine of half opening angle and total amplitude scaling
                cosangle = cos((angleisrc[ii] - angleirec[ii]) / 2.0)
                damp = 2.0 * cosangle * ampisrc[ii] * ampirec[ii] / vel[ii]
                # extract source and receiver angle at given image point
                angle_src = angleisrc[ii]
                angle_rec = angleirec[ii]
                abs_angle_src = abs(angle_src)
                abs_angle_rec = abs(angle_rec)
                aptap = 1.0
                # angle apertures checks
                if (
                    abs_angle_src < angleaperturemax
                    and abs_angle_rec < angleaperturemax
                ):
                    if abs_angle_src >= angleaperturemin:
                        # extract source angle aperture taper value
                        aptap = (
                            aptap
                            * aperturetap[
                                int(
                                    20
                                    * (abs_angle_src - angleaperturemin)
                                    // dangleaperture
                                )
                            ]
                        )
                    if abs_angle_rec >= angleaperturemin:
                        # extract receiver angle aperture taper value
                        aptap = (
                            aptap
                            * aperturetap[
                                int(
                                    20
                                    * (abs_angle_rec - angleaperturemin)
                                    // dangleaperture
                                )
                            ]
                        )
                    # identify x-index of image point
                    iz = ii % nz
                    # aperture check
                    aperture = abs(sixisrcrec - rixisrcrec) / iz
                    if aperture < aperturemax:
                        if aperture >= aperturemin:
                            # extract aperture taper value
                            aptap = (
                                aptap
                                * aperturetap[
                                    int(20 * ((aperture - aperturemin) // daperture))
                                ]
                            )
                        # time limit check
                        if 0 <= itravii < nt - 1:
                            ind1 = (isrc * nr + irec, itravii)
                            val1 = x[ii] * (1 - travdii) * damp * aptap
                            ind2 = (isrc * nr + irec, itravii + 1)
                            val2 = x[ii] * travdii * damp * aptap
                            cuda.atomic.add(y, ind1, val1)
                            cuda.atomic.add(y, ind2, val2)

    @staticmethod
    @cuda.jit
    def _ampsrcrec_kirch_rmatvec_cuda_streams(
        ns,
        nr,
        nt,
        ni,
        nz,
        dt,
        aperturemin,
        aperturemax,
        angleaperturemin,
        angleaperturemax,
        vel,
        aperturetap,
        six_d,
        rix_d,
        travsrc,
        ampsrc,
        anglesrc,
        travi,
        ampi,
        anglei,
        y,
        isrc_list,
        irec,
        x,
    ):
        ii = cuda.grid(1)
        if ni > ii:
            index_isrc = -1
            for isrc in isrc_list:
                index_isrc = index_isrc + 1
                sixisrcrec = six_d[isrc * nr + irec]
                rixisrcrec = rix_d[isrc * nr + irec]
                travirec = travi[:, irec]
                ampirec = ampi[:, irec]
                angleirec = anglei[:, irec]
                travisrc = travsrc[:, isrc]
                ampisrc = ampsrc[:, isrc]
                angleisrc = anglesrc[:, isrc]
                daperture = aperturemax - aperturemin
                dangleaperture = angleaperturemax - angleaperturemin
                trav = travisrc[ii] + travirec[ii]
                itravii = int(trav / dt)
                travdii = trav / dt - itravii
                # extract source and receiver angle at given image point
                angle_src = angleisrc[ii]
                angle_rec = angleirec[ii]
                abs_angle_src = abs(angle_src)
                abs_angle_rec = abs(angle_rec)
                aptap = 1.0

                cosangle = cos((angle_src - angle_rec) / 2.0)
                damp = 2.0 * cosangle * ampisrc[ii] * ampirec[ii] / vel[ii]
                # angle apertures checks
                if (
                    abs_angle_src < angleaperturemax
                    and abs_angle_rec < angleaperturemax
                ):
                    if abs_angle_src >= angleaperturemin:
                        # extract source angle aperture taper value
                        aptap = (
                            aptap
                            * aperturetap[
                                int(
                                    20
                                    * (abs_angle_src - angleaperturemin)
                                    // dangleaperture
                                )
                            ]
                        )
                    if abs_angle_rec >= angleaperturemin:
                        # extract receiver angle aperture taper value
                        aptap = (
                            aptap
                            * aperturetap[
                                int(
                                    20
                                    * (abs_angle_rec - angleaperturemin)
                                    // dangleaperture
                                )
                            ]
                        )
                    # identify x-index of image point
                    iz = ii % nz
                    # aperture check
                    aperture = abs(sixisrcrec - rixisrcrec) / iz
                    if aperture < aperturemax:
                        if aperture >= aperturemin:
                            # extract aperture taper value
                            aptap = (
                                aptap
                                * aperturetap[
                                    int(20 * ((aperture - aperturemin) // daperture))
                                ]
                            )
                        # time limit check
                        if 0 <= itravii < nt - 1:
                            ind1 = ii
                            val1 = (
                                (
                                    x[isrc * nr + irec, itravii] * (1 - travdii)
                                    + x[isrc * nr + irec, itravii + 1] * travdii
                                )
                                * damp
                                * aptap
                            )
                            cuda.atomic.add(y, ind1, val1)

    def _matvec_call(self, *inputs):
        """Handle the forward operation call, dispatching to appropriate CUDA kernels.

        This method selects the appropriate kernel to execute based on the computation flags,
        and performs the forward operation.

        Parameters
        ----------
        *inputs : :obj:`list`
            List of input parameters required by the kernels.

        Returns
        -------
        y : :obj:`numpy.ndarray`
            Output data (seismic data) of forward operation.

        """
        if self.dynamic:
            y_d = self._call_dynamic(inputs[0], "_matvec")
        else:
            y_d = self._call_nondynamic("_matvec", *inputs)
        """
        else:
            x_d = inputs[0]
            y_d = inputs[1]
            ns_d = np.int32(inputs[2])
            nr_d = np.int32(inputs[3])
            nt_d = np.int32(inputs[4])
            ni_d = np.int32(inputs[5])
            dt_d = np.float32(inputs[6])
            trav_srcs_d = to_cupy(inputs[7])
            trav_recs_d = to_cupy(inputs[8])
            self._travsrcrec_kirch_matvec_cuda[
                self.num_blocks, self.num_threads_per_blocks
            ](x_d, y_d, ns_d, nr_d, nt_d, ni_d, dt_d, trav_srcs_d, trav_recs_d)
            cuda.synchronize()
        """
        return y_d

    def _rmatvec_call(self, *inputs):
        """Handle the adjoint operation call, dispatching to appropriate CUDA kernels.

        This method selects the appropriate kernel to execute based on the computation flags,
        and performs the adjoint operation.

        Parameters
        ----------
        *inputs : :obj:`list`
            List of input parameters required by the kernels.

        Returns
        -------
        y : :obj:`numpy.ndarray`
            Output data (image) of adjoint operation.

        """
        if self.dynamic:
            y_d = self._call_dynamic(inputs[0], "_rmatvec")
        else:
            y_d = self._call_nondynamic("_rmatvec", *inputs)
        """
        else:
            x_d = inputs[0]
            y_d = inputs[1]
            ns_d = np.int32(inputs[2])
            nr_d = np.int32(inputs[3])
            nt_d = np.int32(inputs[4])
            ni_d = np.int32(inputs[5])
            dt_d = np.float32(inputs[6])
            trav_srcs_d = to_cupy(inputs[7])
            trav_recs_d = to_cupy(inputs[8])
            self._travsrcrec_kirch_rmatvec_cuda[
                self.num_blocks, self.num_threads_per_blocks
            ](x_d, y_d, ns_d, nr_d, nt_d, ni_d, dt_d, trav_srcs_d, trav_recs_d)
            cuda.synchronize()
        """
        return y_d
