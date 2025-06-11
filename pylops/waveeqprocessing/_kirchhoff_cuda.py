from math import cos, fabs

import numpy as np
from numba import cuda

from pylops.utils.backend import to_cupy


class _KirchhoffCudaHelper:
    """A helper class to perform Kirchhoff demigration/migration using Numba CUDA.

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
        Flag indicating whether to use dynamic computation
        (default is ``False``).

    """

    def __init__(self, ns, nr, nt, ni, dynamic=False):
        self.ns, self.nr, self.nt, self.ni = ns, nr, nt, ni
        self.dynamic = dynamic
        self._grid_setup()

    def _grid_setup(self):
        """Set up CUDA grid and block dimensions.

        This method configures the number of blocks and threads per block for
        CUDA kernels, depending on the number of sources and receivers.
        """
        # use warp size as number of threads per block
        current_device = cuda.get_current_device()
        warp = current_device.WARP_SIZE // 2
        self.num_threads_per_blocks = (warp, warp)
        # configure number of blocks
        self.num_blocks = (
            (self.ns + self.num_threads_per_blocks[0] - 1)
            // self.num_threads_per_blocks[0],
            (self.nr + self.num_threads_per_blocks[1] - 1)
            // self.num_threads_per_blocks[1],
        )

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
                travisrc = trav_srcs[ii, isrc]
                travirec = trav_recs[ii, irec]
                trav = travisrc + travirec
                itravii = int(trav / dt)
                travdii = trav / dt - itravii
                if 0 <= itravii < nt - 1:
                    vii = (
                        x[isrc * nr + irec, itravii] * (1 - travdii)
                        + x[isrc * nr + irec, itravii + 1] * travdii
                    )
                    cuda.atomic.add(y, ii, vii)

    @staticmethod
    @cuda.jit
    def _ampsrcrec_kirch_matvec_cuda(
        x,
        y,
        ns,
        nr,
        nt,
        ni,
        dt,
        vel,
        trav_srcs,
        trav_recs,
        amp_srcs,
        amp_recs,
        aperturemin,
        aperturemax,
        aperturetap,
        nz,
        six,
        rix,
        angleaperturemin,
        angleaperturemax,
        angles_srcs,
        angles_recs,
    ):
        daperture = aperturemax - aperturemin
        dangleaperture = angleaperturemax - angleaperturemin

        isrc, irec = cuda.grid(2)
        if ns > isrc and nr > irec:
            for ii in range(ni):
                sixisrcrec = six[isrc * nr + irec]
                rixisrcrec = rix[isrc * nr + irec]
                travisrc = trav_srcs[ii, isrc]
                travirec = trav_recs[ii, irec]
                trav = travisrc + travirec
                itravii = int(trav / dt)
                travdii = trav / dt - itravii
                ampisrc = amp_srcs[ii, isrc]
                ampirec = amp_recs[ii, irec]
                # extract source and receiver angle at given image point
                angle_src = angles_srcs[ii, isrc]
                angle_rec = angles_recs[ii, irec]
                abs_angle_src = fabs(angle_src)
                abs_angle_rec = fabs(angle_rec)
                # compute cosine of half opening angle and total amplitude scaling
                cosangle = cos((angle_src - angle_rec) / 2.0)
                damp = 2.0 * cosangle * ampisrc * ampirec / vel[ii]
                # identify z-index of image point
                iz = ii % nz
                # angle apertures checks
                aptap = 1.0
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

                    # aperture check
                    aperture = abs(sixisrcrec - rixisrcrec) / (iz + 1)
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
    def _ampsrcrec_kirch_rmatvec_cuda(
        x,
        y,
        ns,
        nr,
        nt,
        ni,
        dt,
        vel,
        trav_srcs,
        trav_recs,
        amp_srcs,
        amp_recs,
        aperturemin,
        aperturemax,
        aperturetap,
        nz,
        six,
        rix,
        angleaperturemin,
        angleaperturemax,
        angles_srcs,
        angles_recs,
    ):
        daperture = aperturemax - aperturemin
        dangleaperture = angleaperturemax - angleaperturemin

        isrc, irec = cuda.grid(2)
        if ns > isrc and nr > irec:
            for ii in range(ni):
                sixisrcrec = six[isrc * nr + irec]
                rixisrcrec = rix[isrc * nr + irec]
                travisrc = trav_srcs[ii, isrc]
                travirec = trav_recs[ii, irec]
                trav = travisrc + travirec
                itravii = int(trav / dt)
                travdii = trav / dt - itravii
                ampisrc = amp_srcs[ii, isrc]
                ampirec = amp_recs[ii, irec]
                # extract source and receiver angle at given image point
                angle_src = angles_srcs[ii, isrc]
                angle_rec = angles_recs[ii, irec]
                abs_angle_src = fabs(angle_src)
                abs_angle_rec = fabs(angle_rec)
                # compute cosine of half opening angle and total amplitude scaling
                cosangle = cos((angle_src - angle_rec) / 2.0)
                damp = 2.0 * cosangle * ampisrc * ampirec / vel[ii]
                # identify z-index of image point
                iz = ii % nz
                # angle apertures checks
                aptap = 1.0
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

                    # aperture check
                    aperture = abs(sixisrcrec - rixisrcrec) / (iz + 1)
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

    def _call_kinematic(self, opt, *inputs):
        """Kinematic-only computations using CUDA.

        This method handles data preparation and execution of CUDA kernels
        for both forward and adjoint operations of kinematic-only operator.

        Parameters
        ----------
        opt : :obj:`str`
            Operation type, either '_matvec' for forward or '_rmatvec'
            for adjoint.
        *inputs : :obj:`list`
            List of input parameters required by the kernels.

        Returns
        -------
        y_d : :obj:`cupy.ndarray`
            Output data.

        """
        x_d = inputs[0]
        y_d = inputs[1]
        ns_d = np.int32(inputs[2])
        nr_d = np.int32(inputs[3])
        nt_d = np.int32(inputs[4])
        ni_d = np.int32(inputs[5])
        dt_d = np.float32(inputs[6])
        trav_srcs_d = to_cupy(inputs[7])
        trav_recs_d = to_cupy(inputs[8])

        if opt == "_matvec":
            self._travsrcrec_kirch_matvec_cuda[
                self.num_blocks, self.num_threads_per_blocks
            ](x_d, y_d, ns_d, nr_d, nt_d, ni_d, dt_d, trav_srcs_d, trav_recs_d)
        elif opt == "_rmatvec":
            self._travsrcrec_kirch_rmatvec_cuda[
                self.num_blocks, self.num_threads_per_blocks
            ](x_d, y_d, ns_d, nr_d, nt_d, ni_d, dt_d, trav_srcs_d, trav_recs_d)
        cuda.synchronize()

        return y_d

    def _call_dynamic(self, opt, *inputs):
        """Synamic computations using CUDA.

        This method handles data preparation and execution of CUDA kernels
        for both forward and adjoint operations of dynamic operator.

        Parameters
        ----------
        opt : :obj:`str`
            Operation type, either '_matvec' for forward or '_rmatvec'
            for adjoint.
        *inputs : :obj:`list`
            List of input parameters required by the kernels.

        Returns
        -------
        y_d : :obj:`cupy.ndarray`
            Output data.

        """
        x_d = inputs[0]
        y_d = inputs[1]
        ns_d = np.int32(inputs[2])
        nr_d = np.int32(inputs[3])
        nt_d = np.int32(inputs[4])
        ni_d = np.int32(inputs[5])
        dt_d = np.float32(inputs[6])
        vel_d = to_cupy(inputs[7])
        trav_srcs_d = to_cupy(inputs[8])
        trav_recs_d = to_cupy(inputs[9])
        amp_srcs_d = to_cupy(inputs[10])
        amp_recs_d = to_cupy(inputs[11])
        aperturemin_d = np.float32(inputs[12])
        aperturemax_d = np.float32(inputs[13])
        aperturetap_d = to_cupy(inputs[14])
        nz_d = np.int32(inputs[15])
        six_d = to_cupy(inputs[16])
        rix_d = to_cupy(inputs[17])
        angleaperturemin_d = np.float32(inputs[18])
        angleaperturemax_d = np.float32(inputs[19])
        angles_srcs_d = to_cupy(inputs[20])
        angles_recs_d = to_cupy(inputs[21])

        if opt == "_matvec":
            self._ampsrcrec_kirch_matvec_cuda[
                self.num_blocks, self.num_threads_per_blocks
            ](
                x_d,
                y_d,
                ns_d,
                nr_d,
                nt_d,
                ni_d,
                dt_d,
                vel_d,
                trav_srcs_d,
                trav_recs_d,
                amp_srcs_d,
                amp_recs_d,
                aperturemin_d,
                aperturemax_d,
                aperturetap_d,
                nz_d,
                six_d,
                rix_d,
                angleaperturemin_d,
                angleaperturemax_d,
                angles_srcs_d,
                angles_recs_d,
            )
        elif opt == "_rmatvec":
            self._ampsrcrec_kirch_rmatvec_cuda[
                self.num_blocks, self.num_threads_per_blocks
            ](
                x_d,
                y_d,
                ns_d,
                nr_d,
                nt_d,
                ni_d,
                dt_d,
                vel_d,
                trav_srcs_d,
                trav_recs_d,
                amp_srcs_d,
                amp_recs_d,
                aperturemin_d,
                aperturemax_d,
                aperturetap_d,
                nz_d,
                six_d,
                rix_d,
                angleaperturemin_d,
                angleaperturemax_d,
                angles_srcs_d,
                angles_recs_d,
            )
        cuda.synchronize()

        return y_d

    def _matvec_cuda(self, *inputs):
        """Forward with dispatching to appropriate CUDA kernels.

        This method selects the appropriate kernel to execute based on the
        computation flags, and performs the forward operation.

        Parameters
        ----------
        *inputs : :obj:`list`
            List of input parameters required by the kernels.

        Returns
        -------
        y_d : :obj:`cupy.ndarray`
            Output (seismic data) of the forward operator.

        """
        if self.dynamic:
            y_d = self._call_dynamic("_matvec", *inputs)
        else:
            y_d = self._call_kinematic("_matvec", *inputs)

        return y_d

    def _rmatvec_cuda(self, *inputs):
        """Adjoint with dispatching to appropriate CUDA kernels.

        This method selects the appropriate kernel to execute based on the
        computation flags, and performs the adjoint operation.

        Parameters
        ----------
        *inputs : :obj:`list`
            List of input parameters required by the kernels.

        Returns
        -------
        y_d : :obj:`cupy.ndarray`
            Output data (image) of the adjoint operator.

        """
        if self.dynamic:
            y_d = self._call_dynamic("_rmatvec", *inputs)
        else:
            y_d = self._call_kinematic("_rmatvec", *inputs)

        return y_d
