function cat24_bcs(job2)
  % cat24 bias correction
  % - I tested our Bestoff sample and some others
  % - It should make no big difference in the resulting quality 
  %   which space (native/resampled) or if denoising was used 
  % - Runtime grows a bit stronger for higher resolutions, e.g., 
  %     IXI ~ 10s, HR075 ~ 30s, yv98_mprage_05mm ~ 86s
  %
  % - subcortical structures could be better >> use atlas input
  
  job.verb             = 1; % display progress (0-none,1-subject,2-details)
  job.dt               = 2; % SPM image datattype (16  - float, 512 - int16, 2 - uint8)
  job.normInt          = 1; % intensity normalization 
  job.simpleApprox     = 2; % use simple approximation rather than cat_vol_approx #########################
  job.write.org        = 0; % write intensity normed image (for debugging)
  job.write.GAS        = 0; % write global adapted segmentation (GAS) map 
  job.write.LAS        = 1; % do and write local adapted segmentation (LAS) map
                            % this is a bit inoptimal for the histogram but helps visually
                            % however, in some protocols it could be also helpful (eg. MP2Rage) 
  job.write.lasseg     = 1; % do and write fast PVE segmentation                            
  job.write.biasfield  = 1; % write bias field map (for debugging) 
  
  if exist('job2','var')
    job = cat_io_checkinopt(job2,job); 
    P = job.files; 
  else
    % #### just my manual test case - remove later ####
    P = {
      '01_Bestoff/4397_T1_thinWM'                                   % 1 - could be better
      '01_Bestoff/Aarhus_TorbenLund_R1_0001'                        % 2  
      '01_Bestoff/ADHD200NYC_T1_defacedMoveArt'                     % 3
      '01_Bestoff/ADNI_AD_100_0743_T1_sd000000-rs00_S17224_I62363'  % 4
      '01_Bestoff/ADNI_AD_114_0228_T1_sd000000-rs00_S11697_I49735'  % 5 - BG sphere issue !!!
      '01_Bestoff/BUSS02_T1_infant'                                 % 6 - subcortical?
      '01_Bestoff/Cathy6_T1_hydrocephlus'                           % 7 - UPSI ... !!!
      '01_Bestoff/HR075_MPRAGE'                                     % 8
      '01_Bestoff/Hendrik_FLAIR'                                    % 9
      '01_Bestoff/HR075_T2'                                         % 10 - subcortical in Yml worse
      '01_Bestoff/Magdeburg_TilmannKlein_S26_mp2rage'               % 11
      '01_Bestoff/OASIS1_AD_001_0031_T1_sd000000-rs00'              % 12
      '01_Bestoff/SRS_R0500my_Tohoku_VBM8bo'                        % 13
      ...'01_Bestoff/Leipzig_Bazin_MP2RAGE_2017_194140_T1w'            % 14 missing label
      '01_Bestoff/IXI_HC_HH_175_T1_SD000000-RS00'                   % 15
      '01_Bestoff/IXI_HC_IO_543_T1_SD000000-RS00'                   % 16
      ...
      '04_MR_Protocols/yv98_mprage_05mm'                            % 17
      '04_MR_Protocols/Magdeburg7T_skc73'                           % 18
      '04_MR_Protocols/Gorgolewski2017_T1_7T'                       % 19
      '04_MR_Protocols/IXI175-HH-1570-MRA'                          % 20
      '04_MR_Protocols/London_GabrielZiegler_20160303_NSPNID_10736_00_MT' % 21
      '04_MR_Protocols/London_GabrielZiegler_20160303_NSPNID_10736_00_R1' % 22
      '04_MR_Protocols/rsub-bhl522_T1w_rs'
      '04_MR_Protocols/Gorgolewski2017_T2_7T'
      '04_MR_Protocols/FlaviaArduini_008_t2_tse_sag_volumetrica_iso'
      '04_MR_Protocols/Colin27_T2'
      '04_MR_Protocols/Colin27_T1'
      '04_MR_Protocols/Colin27_PD'
      '04_MR_Protocols/Cathy0_T1_MP2U'
      '04_MR_Protocols/Cathy0_T1_MP2'
      '04_MR_Protocols/Cathy0_T1_MP1'
      '04_MR_Protocols/Cathy0_PD_MP2'
      '04_MR_Protocols/Hanke2015_T2_7T'
      '04_MR_Protocols/Hanke2015_T1_7T'
      '04_MR_Protocols/yv98_05mm_corrected_division_n3' % CSF-LAS-bc
      };
  end
  six   = 1:numel(P);  % subjects
  
  
  %%
  for si = 3 %six(1:end)     
    % si=1;
  
    stime2 = cat_io_cmd(spm_str_manip(P{si},'a45'),'g9','',job.verb);  
    stime  = cat_io_cmd('  Prepare data','g5','',job.verb-1,stime);  
    
    % input filenames
    if exist('job2','var')
      Po  = spm_file(P{si},'suffix','_desc-sanlm');
      Pl  = spm_file({si} ,'suffix','_desc-high_label');
      Pa  = spm_file(P{si},'suffix','_desc-high_atlas');
    else
      Po  = sprintf('/Users/rdahnke/Desktop/DeepCAT/%s_desc-sanlm.nii', P{si});
      Pl  = sprintf('/Users/rdahnke/Desktop/DeepCAT/%s_res-high_label.nii', P{si}); 
      Pa  = sprintf('/Users/rdahnke/Desktop/DeepCAT/%s_res-high_atlas.nii', P{si});
    end
    % output filenames 
    Po2 = spm_file(Po,'suffix','_org');
    Pm  = spm_file(Po,'suffix',sprintf('_bias_sA%d', job.simpleApprox)); 
    Pml = spm_file(Po,'suffix',sprintf('_las_sA%d' , job.simpleApprox)); 
    % the bias field maps are for debugging
    Pms = spm_file(Po,'suffix',sprintf('_laslabel_sA%d' , job.simpleApprox));
    Pbf = spm_file(Po,'suffix',sprintf('_biasfield_sA%d', job.simpleApprox)); 
    Plf = spm_file(Po,'suffix',sprintf('_lasfield_sA%d',  job.simpleApprox)); 
  
    % TEMPORARY if the denoised file is missing than use the original one
    if ~exist(Po,'file')
      Po  = sprintf('/Users/rdahnke/Desktop/DeepCAT/%s.nii', P{si});
    end
  

    % load header and images depending on the image space
    native = 1; % 0 - normalized label maps space, 1 - native space of the Yo 
    Vo  = spm_vol(Po); 
    Vl  = spm_vol(Pl);
    Va  = spm_vol(Pa);
    if native
      Yo  = single(spm_read_vols(Vo)); 
      Vp0 = Vo; Vp0.fname = spm_file(Vp0.fname,'suffix','_seg-native');
      [~,Yp0] = cat_vol_imcalc([Vo,Vl],Vp0,'i2');
      Va0 = Vo; Va0.fname = spm_file(Va0.fname,'suffix','_atlas-native');
      [~,Ya]   = cat_vol_imcalc([Vo,Va],Va0,'i2',struct('interp',0));
    else
      Yp0 = spm_read_vols(Vl); 
      Ya  = spm_read_vols(Va); 
      Vp0 = Vp0; Vp0.fname = spm_file(Vo.fname,'suffix','_org-mri');
      [Vo,Yo] = cat_vol_imcalc([Vl,Vo],Vp0,'i2');
    end
  
  
    % ### Subject / image information ### 
    % These are some simple basic information about the image and subject. 
    % get single tissue class from label map (CSF=1, GM=2, WM=3)
    Yp0toC = @(Yp0,c) max(0,1-min(1,abs(Yp0-c)));
    vx_vol = sqrt(sum(Vo.mat(1:3,1:3).^2));   
  
  
    % Gradient edge map with average tissue intensity (eg. double it to threshold)
    % Edge maps can be easily normalized Yg./T3th or Yg./Yo and are then bias
    % free because of their local nature!), as the normalization can change
    % it is not inlduced here yet. The mimimum is used to avoid aging bias, 
    % with more 'noisy' WM by perivescular spaces in elderly but at once huge 
    % ventricles that are on the other side not available in young subjects. 
    % Nevertheless, we will try to avoid outliers with "Yg(:)./Yo(:)<.5".
    T3th  = [cat_stat_nanmedian( Yo( Yp0toC(Yp0(:),1)>.95)) ...
             cat_stat_nanmedian( Yo( Yp0toC(Yp0(:),2)>.95)) ...
             cat_stat_nanmedian( Yo( Yp0toC(Yp0(:),3)>.95))];
    bias  = cat_stat_nanstd( Yo( Yp0toC(Yp0(:),3)>.9 )) ./ min(abs(diff(T3th)));
    Yg    = cat_vol_grad(Yo) .* cat_vol_morph(Yo~=0,'e',1);
    Yg    = Yg ./ max( Yo , max( max(T3th)/5 , min(T3th)/2 )); %* (0.25/bias)
    gth   = max( 0.05 , min( ...
             [ cat_stat_nanmedian( Yg( Yp0(:)==0 & Yg(:)<cat_stat_nanmedian(Yg(Yp0(:)==0)) )) , ... % background
               cat_stat_nanmedian( Yg( Yp0toC(Yp0(:),1)>.95 & Yg(:)<.9)) , ... % CSF
               cat_stat_nanmedian( Yg( Yp0toC(Yp0(:),3)>.95 & Yg(:)<.9)) ] )); % WM
   
    % Estimate tissue thresholds of major structures to avoid bias by
    % missclassfied objects and larger PVE areas.  
    Ywm   = cat_vol_morph( Yp0toC(Yp0,3)>.9 & Yg<gth*2 , 'l' , [10 0.1]) > 0; 
    Ygm   = Yp0toC(Yp0,2)>.9 & ~cat_vol_morph( Yp0toC(Yp0,2)>.5 ,'o',3); 
    Ycm   = cat_vol_morph( Yp0toC(Yp0,1)>.9 & Yg<gth*2 , 'lo' , 1);
    if sum(Ycm)<1000, Ycm = Yp0toC(Yp0,1)>.9; end % just in case the threshold is to agressive
    T3th  = [cat_stat_nanmedian( Yo( Ycm(:) )) ...
             cat_stat_nanmedian( Yo( Ygm(:) )) ...
             cat_stat_nanmedian( Yo( Ywm(:) ))];
    
    % estimate image contrast (T1w/T2w/PDw) 
    if     T3th(1) < T3th(2) && T3th(2) < T3th(3), modality = 1;   % T1w
    elseif T3th(1) > T3th(2) && T3th(2) > T3th(3), modality = 2;   % T2w
    elseif T3th(1) < T3th(3) && T3th(2) < T3th(3), modality = 0;   % PDw - WM is maximum
    else                                         , modality = 3;   % PDw - WM is minimum
    end
  
  
    % Fast bias correction and background estimation
    % Some inital very low-frequent tissue unspecific bias-correction that we 
    % need for masking and especially for the general correction in the head
    % and backround.
    stime = cat_io_cmd('  Initial bias correction','g5','',job.verb-1,stime); 
    Yp0c  = Yp0; 
    
    % remove objects within the CSF that we are not useful to  ####
    Yp0c(Yp0c<1) = 0; % remove low properbility CSF (skull, meninges, blood-vessels, ...)
    Yp0c(Yp0>=1 & Yp0<2 & Yg>gth*4) = 0; % meninges, blood-vessels, ...

    % remove difference between the segmentation and real edges (eg. 4397)
    Ygp0 = cat_vol_grad(Yp0/3);
    Yp0c = Yp0c .* (abs(Ygp0 - Yg)<gth*4); 
    Yp0c(smooth3(Yp0c>0)<.5) = 0;


    %% create simulated tissue map Yi to estimate the bias Yw
    [Ygr,Yor,Yp0r,Yp0cr,rs] = cat_vol_resize( {Yg,Yo,Yp0,Yp0c} , 'reduceV',vx_vol,1.2,16,'meanm');;
 
    % brain tissue
    Yp0t = zeros(size(Yp0cr)); for ci = 1:3, Yp0t = Yp0t + T3th(ci) .* Yp0toC(Yp0cr,ci); end
    % head and mp2rage background in general
    Yhd  = smooth3( Yor>min( max(0.1,0.1*(1/bias)) * max(T3th), median(T3th))  &  cat_vol_morph(Yp0r==0,'e',2)  &  Ygr<gth*8 )>.5;
    % high intensity mp2rage background
    Ymp2 = smooth3( Yhd & Ygr<0.4 & Ygr~=0 & ...
           (Yor>0.9 * cat_stat_nanmedian(Yor(Yhd(:)))) & ...
           (Yor<1.1 * cat_stat_nanmedian(Yor(Yhd(:)))) )>.5;
    % low intensity head tissue (muscles)
    Yhdl = Ygr<gth*4 & Ygr~=0 & ~Ymp2 & Yhd & Yor>.1/max(T3th) & Yor<max(T3th); Yhdl(smooth3(Yhdl)<.5) = 0;
    % high intensity head tissue (fat)
    Yhdh = Ygr<gth*4 & Ygr~=0 & ~Ymp2 & Yhd & ~Yhdl; Yhdh(smooth3(Yhdh)<.5) = 0;
    % combine all brain and head tissues
    Yi = Yp0t  ...
         +  Yhdl * max([cat_stat_nanmedian([0;Yor(Yhdl(:))]), median(T3th) / (1+(modality>=2)) ]) ... % high intensity head (fat)
         +  Yhdh * max([cat_stat_nanmedian([0;Yor(Yhdh(:))]), max(T3th)    / (1+(modality>=2)) ]) ... % low intensity head (muscles)
         +  Ymp2 * cat_stat_nanmedian([0;Yor(Ymp2(:))]);     % mp2rage background 
    Yi   = max(0.01*T3th(3),Yor) ./ max(Yi,eps) .* (Yi>0);   % negative values could be critial (e.g interpolation artifacts in CSF )
  
    %% full field approximation
    if job.simpleApprox == 1
      Ywr = simple_approx( Yi , min(10,max(2,1 / bias)) , 3 , 2); 
    elseif job.simpleApprox == 2
      Ywr = simple_approx2( Yi, 3 - mean( rs.vx_red - 1) , 10); % very smooth3 background
    else
      Ywr = cat_vol_approx( Yi, 'nn' , vx_vol, 2);
      Ywr = cat_vol_smooth3X( Ywr , 4); 
    end
    Yw    = cat_vol_resize( Ywr , 'dereduceV', rs);

    % apply bias correction and restore skull-stripping/defacing values
    Ym = Yo ./ Yw; 
    Ym = Ym ./ cat_stat_nanmedian(Ym(Ywm(:)));
    Ym(smooth3(Yo)==0) = 0; 
    

  
  
    %% (3) Local Adaptive Segmenation (LAS)
    %  ----------------------------------------------------------------------
    %  This function estimates the local tissue intensity based within a masked
    %  area (eg. the WM), given by a standard tissue segmentation, and approximates
    %  values outside the max and smoothes values within the mask. 
    %  - This part is already working quite well -
    if job.write.LAS || job.write.lasseg 
      stime = cat_io_cmd('  Local Adaptive Segmenation (LAS)','g5','',job.verb-1,stime); 

      % update thresholds
      %csfth = cat_stat_kmeans( Ym( Ycm(:) ),2,1); 
      T3thm = [cat_stat_nanmedian( Ym( Ycm(:) )) ... csfth(1) ... 
               cat_stat_nanmedian( Ym( Ygm(:) )) ...
               cat_stat_nanmedian( Ym( Ywm(:) ))];

      % estimate bias for each tissue class
      Ylab   = cell(1,6);   
      if job.normInt, T3thmn = [0 1 2 3 4 6] / 3; % with normalization
      else, T3thmn = [0 T3thm T3thm(3)+diff(T3thm(2:3)) max(Ym(:))]; 
      end 
      Yp0t = zeros(size(Yp0)); for ci = 1:3, Yp0t = Yp0t + T3thm(ci) .* Yp0toC(Yp0,ci); end
      [~,T3thms] = sort(T3thm,'ascend'); 
      for ci = 3:-1:1 
        % extract tissue intensity (major class, avoid PVE regions, avoid edges)
        Yi = Yp0toC(Yp0,T3thms(ci)) > 0.5  &  abs(Ym-Yp0t) < 0.15  &  Yg<gth*4 * (1 + (modality>1));
        Yi(smooth3(Yi)<0.5) = 0; % remove small (instable) regions 
        Yi = Yi .* Ym;
        %%
        if 0 % maybe good / maybe not required and quite slow
          switch round(ci)
            case 1, Yi = cat_vol_localstat(Yi,Yi>0,1,2);
            case 2, Yi = cat_vol_localstat(Yi,Yi>0,1,1);
            case 3, Yi = cat_vol_localstat(Yi,Yi>0,1,3);
          end
        end

        % add background
        Yi = Yi  +  (Yp0==0) * T3thm(T3thms(ci)); 

        % approximation of unmasked values with smoother tissues
        if job.simpleApprox == 1
          Ylab{ci+1} = simple_approx( Yi , 1 * (1 + (modality>1)), 3 , 4); 
        elseif job.simpleApprox == 2
          Ylab{ci+1} = simple_approx2( Yi , 2 / mean(vx_vol) , 60); 
        else
          Ylab{ci+1} = cat_vol_approx( Yi , 'nn' , vx_vol, 4); 
        end

        % avoid crossing (eg. in low quality images like FLAIR)
        if ci<3, Ylab{ci+1} = min( Ylab{ci+1}, 0.9*Ylab{ci+2}); end
      end
      Ylab{6} = max(Ym(:)); 
      Ylab{5} = Ylab{4} + (Ylab{4} - Ylab{3});
      Ylab{2} = max(0.01,Ylab{2}); 
      Ylab{1} = min(0,median(Ylab{2}(:)) - 2 * std(Ylab{2}(:)));  
   

      % Step-wise intensity normalization starting with the highes intensity tisuse. 
      % Although this kind of scaling is not optimal for the histogram it allows
      % much simpler use e.g. to estimate the differences to a (label) segmentation.
      limap = @(Yml,ci) Yml  +  (Yml==0)  .*  ( (Ym>=Ylab{ci} & Ym<Ylab{ci+1} )  .* ... 
        (T3thmn(ci) + (Ym-Ylab{ci}) ./ max(eps,Ylab{ci+1}-Ylab{ci}) .* max(eps,T3thmn(ci+1)-T3thmn(ci)) ));
     
      Yml = zeros(size(Ym)); 
      for ci = numel(Ylab)-1:-1:1, Yml = limap(Yml, ci); end 
      Ym(smooth3(Yo)==0) = 0; 
    end
  
  
    % #### denoising in intensity normalized data! ####
  
  
    
  
    %% write output (more for debugging)
    %  ----------------------------------------------------------------------
    %  - to save some disk space we use int16 (range: -32768:1:32767)
    %  - although we scale images with lowest values in average to zero, 
    %    negative values are still possible and uint16 would just cut them
    %  - to have some simple image intensity range we set the WM to 100 and
    %    use 2 digits (pinfo 1e-2) to have some more details resulting in 
    %    a data range of -327.68:0.01:327.67 
    %  - uint16 would give no advantage here and we could also easily use 
    %    int8 (-128:1:127) or uint8(0:1:255) where this this time the uint8
    %    is more save in case of non-T1 modalities
    %  - the biasfield is only relevant for tests and to see if we get some
    %    smooth field without structural aspects, ie. if you see the head or 
    %    brain structures then the correction was not good
    stime = cat_io_cmd('  Write output','g5','',job.verb-1,stime); 
    switch job.dt
      case 512, pinfo=1e-2;                     % uint16
      case 2,   pinfo=1 / (1 + (modality<2));   % uint8
      case 16,  pinfo=1;                        % single
    end 
    Vm  = Vo;  Vm.fname = Po2;  Vm.dt(1) = job.dt;  Vm.pinfo(1) = pinfo; 
    
    % original map but intensity normalized for fast comparison
    % - in case of T2w we need to integrate the ultra high CSF values 
    if job.write.org
      Vm.fname = Po2;  
      Yo2 = Yo ./ (cat_stat_nanmedian(Yo(Yp0c>2.9 & Yp0>2.9)) * (1 + (modality==2)));
      spm_write_vol(Vm, max(-1,min(4,Yo2)) * 100);
      clear Yo2; 
    end
    
    % normalized images
    if job.write.GAS == 1  ||  job.write.GAS == 3
      Vm.fname = Pm;  
      spm_write_vol(Vm, max(-1,min(4,Ym)) * 100);
    elseif job.write.GAS == 2  ||  job.write.GAS == 3
      Vm.fname = Pm;    
      spm_write_vol(Vm, max(-1,min(4,Ym)) * 100);
    end
    
    % local adaptive segmentation 
    % #### better limites for MT! and type of normalization  ####   
    if job.write.LAS
      Vm.fname = Pml;   
      spm_write_vol(Vm, max(0,min(1.5 + 2*(modality>1),Yml)) * 100);
    end
    if job.write.lasseg
      %% write simple segmentation with PVE class
      [~,T3ths] = sort(T3th);  T3ths(4) = 3; T3ths = [0 T3ths];
      Yp0t = zeros(size(Yp0),'single'); for ci = 1:5, Yp0t = Yp0t + T3ths(ci) .* Yp0toC(1 + Yml*3,ci); end
      if modality==1, Ymx = Yml; else, Ymx = Yp0t; end
      Yb  = cat_vol_smooth3X( Yp0>.5 , 4 )>.49;   
      Ymx = Ymx .* Yb; % skull-stripping
      Ymx = max(Yb*.5, min(3,round(Ymx * 6 / (1 + 2*(modality>1)) ) / 2)); 
      Ymx = min(Yb + 2 * max(Yp0>2,smooth3(cat_vol_morph(smooth3(Ymx)>1.45,'ldo',1))) ,Ymx); 
      Vms = Vo;  Vms.fname = Po2;  Vms.dt(1) = 2;  Vms.pinfo(1) = 1e-1; Vms.fname = Pms;   
      spm_write_vol(Vms, Ymx);
    end
    
    % bias field map
    if job.write.biasfield
      Vm.fname = Pbf;   
      Ywx = Yw / cat_stat_nanmedian(Yw(Ywm(:))); 
      Ywx(:,:,1) = 1/1.5; Ywx(:,:,end) = 1.5;       % add scaling bars
      spm_write_vol(Vm, max(0,min(4,Ywx)) * 100); clear Ywx
    end
  
    % #### LAS correction map ####
    if job.write.biasfield && 1
      %%
      Vm.fname = Plf;  
      if 0
        Ywx = ( Ylab{2}/mean(Ylab{2}(Yp0(:)>0)) ...
              + Ylab{3}/mean(Ylab{3}(Yp0(:)>0)) ...
              + Ylab{4}/mean(Ylab{4}(Yp0(:)>0)) ) / 3; 
      else
        % minimal contrast 
        Ywx = min( cat( 4 , ...
          1 - 3   * abs(Ylab{2}/mean(Ylab{2}(Yp0(:)>0)) -  Ylab{3}/mean(Ylab{3}(Yp0(:)>0)) ) , ...
          1 - 3   * abs(Ylab{3}/mean(Ylab{3}(Yp0(:)>0)) -  Ylab{4}/mean(Ylab{4}(Yp0(:)>0)) ) , ...
          1 - 3/2 * abs(Ylab{2}/mean(Ylab{2}(Yp0(:)>0)) -  Ylab{4}/mean(Ylab{4}(Yp0(:)>0)) ) ) , [] , 4 );
      end
      Ywx = cat_vol_smooth3X(Ywx,4); % the map combines multiple effects that result in some not inoptimal edges
      Ywx(:,:,1) = 0.5; Ywx(:,:,end) = 1.25; 
      spm_write_vol(Vm, max(0,min(4,Ywx)) * 100);
    end
  
    if job.verb>1, cat_io_cmd('  ','g5','',job.verb-1,stime); end
    if job.verb, fprintf('%5.0fs\n',etime(clock,stime2)); end
  end
  fprintf('done.\n')
end
function Ya = simple_approx(Y,s1,s2,red,Ymsk)
%simple_approx. Simple approximation by the closest Euclidean value.
%
%  [Yo,D,I] = simple_approx(Y[,s,Ymsk])
%  Y    .. input  image those zeros will be approximated by closes values
%  Ya   .. output image with appoximated values 
%  s1   .. smoothing filter size (default = 1)
%  s2   .. smoothing filter size for distant values (defaut = 10)
%  Ymsk .. limit approximation (default = full image)
%

  if ~exist('s1', 'var'), s1 = 0.5; end
  if ~exist('s2', 'var'), s2 = 5; end
  if ~exist('Ymsk', 'var'), Ymsk = true(size(Y)); end
  if ~exist('red', 'var'), red = 4; end
  
  % use lower resolution for faster processing 
  if red > 1
    [Y,Ymsk,res] = cat_vol_resize( {Y,single(Ymsk)} , 'reduceV',1,red,16,'meanm');
    Ymsk = Ymsk > 0.5;
    s1 = s1/red;
    s2 = s2/red;
  end

  % this reduces the PVE
  Y = cat_vol_median3(Y,Y~=0,Y~=0);
  Y = cat_vol_localstat(Y,Y~=0,1,1,4);

  % estimate closest object point 
  [D,I] = cat_vbdist(single(Y~=0),Ymsk > 0); 
  
  % align (masked) non-object voxels with closest object value
  Ya = Y(I); 

  % smooth the result - correct for average mnYo to avoid smoothing boundary issues
  D = max(0,min(1,(D - s2/2) / s2)); 
  mnYo = cat_stat_nanmedian(Ya(Ya(:)~=0)); Ya = Ya - mnYo; Ya(~Ymsk) = 0; 
  Yas  = Ya; spm_smooth(Yas  , Yas  , repmat( double(s1) ,1,3)); 
  Yas2 = Ya; spm_smooth(Yas2 , Yas2 , repmat( double(s2),1,3)); 

  % keep the global variance after smoothing  
  Yas  = Yas  / cat_stat_nanstd(Yas(:))  * cat_stat_nanstd(Ya(:));
  Yas2 = Yas2 / cat_stat_nanstd(Yas2(:)) * cat_stat_nanstd(Ya(:));
  
  % combine the less smooth local map with a smoother map for far distant regions 
  Ya   = Yas .* (1-D) + Yas2 .* D;

  % recorrect for the average
  Ya = Ya + mnYo; Ya(~Ymsk) = 0; 

  % back to original resolution
  if red>1
    Ya = cat_vol_resize( Ya , 'dereduceV', res);
    spm_smooth(Ya, Ya, repmat(red/2,1,3));
  end
end
function Ya = simple_approx2(Y,rec,s)
%simple_approx2. Simple approximation of missing values. 
% In a given image all "abnormal" values, i.e. nans, infs and zeros, are 
% replace by "normal" values. Zeros are defined as abnormal to a allow  
% simple use of masks. 
% For each abnormal value the closest normal value is used. The result is
% strongly filtered (controlled by the s parameter with default = 30). For
% performance the opperation is carried out interatively on half resolutions 
% controled by the rec parameter (default = 3). 
%
%  Ya = simple_approx(Y[,rec,s])
%  Y    .. input  image those zeros will be approximated by closes values
%  Ya   .. output image with appoximated values
%  rec  .. number of recursive calls with half resolution (default = 3) 
%          i.e an image with 256 voxel will be reduced to 128, 64, and 
%          finally 32 voxels (minimum matrix size is 8 voxels)
%  s    .. smoothing filter size (default = 30)
%
% TODO: The map looks more or less smooth but this depends less on
% smoothing and more on the number of reductions that should ideally only 
% control computational speed. 
% >> It is maybe useful to integrate also the voxel-size and some gaussian 
%    like smoothing to support simpler control or more expected results.

  if ~exist('s',    'var'), s    = 30; else, s   = double(s);  end
  if ~exist('rec',  'var'), rec  = 3;  else, rec = round(rec); end
 
  % intensity normalization of the input and set special values
  mdY  = cat_stat_nanmedian(Y(Y(:)~=0)); % / 100; 
  Yc   = Y ./ mdY; Yc(isnan(Yc)) = 0; Yc(isinf(Yc) & Yc<0) = 0; Yc(isinf(Yc) & Yc>0) = 0;  
 
  % use lower resolution for faster processing 
  [Yr,res] = cat_vol_resize( Yc, 'reduceV', 1, 1 + (rec>0), 8, 'meanm');

  % iteratively lower/half resolution 
  if rec > 0, Ya = simple_approx2(Yr, rec - 1, s / mean(res.vx_red)); end
  
  if ~exist('Ya','var')
  % approximation routine on the lowest resolution level
  
    % remove outiers
    Yr = cat_vol_median3(Yr,Yr~=0,Yr~=0); 
    Yr = cat_vol_localstat(Yr,Yr~=0,1,1); 

    % estimate closest object point 
    [D,I] = cat_vbdist(single(Yr~=0)); %D = D./10;
    % align (masked) non-object voxels with closest object value
    Ya = Yr(I); 
    
    % remove outliers
    Ya = cat_vol_median3(Ya,Yr==0,Yr==0);

    % background filtering 
    for ci=0.5:-0.1:0.1
      s2 = min(10^-1, max(10^-5, 1 / s * 1000)); % filter limits
      Ya = cat_vol_laplace3R(Ya , D>ci, s2 ); % smooth original values
    end

    % The distance-based mapping only repeats values and also the smoothing 
    % reduces the amount of corrections espeicially in the background that
    % we smoothing strong ... the power operations compensate this a bit. 
    Ya = Ya.^min(1.5,1 + D); 
  else
    % replace strong outliers by filtered
    bias = std(Yr(:)) / 1;
    Ya = Ya ./ cat_stat_nanstd(Ya(Ya(:)>0)) * cat_stat_nanstd(Ya(Ya(:)>0));
    Yr(  Yr<Ya / (1+bias) | Yr>Ya * (1+bias) ) = 0; 
    Ya(Yr>0) = Yr(Yr>0);
  end

  % filter
  s2 = min(10^-1, max(10^-2, .1 / s));            % filter limits
  Ya = cat_vol_laplace3R(Ya, Yr==0         , s2); % keep original values 
  Ya = cat_vol_laplace3R(Ya, true(size(Yr)), s2); % smooth original values

  % correct for changes by smoothing  
  Ya = min(10, max( 0.1, Ya / cat_stat_nanmean(Ya(Yr(:)>0)))) * cat_stat_nanmean(Yr(Yr(:)>0));
  
  % back to original resolution
  Ya = cat_vol_resize( Ya , 'dereduceV', res);

  % descale and reset special values
  Ya = Ya .* mdY; Ya(isnan(Y)) = nan; Ya(isinf(Y) & Y<0) = -inf; Ya(isinf(Y) & Y>0) = inf; 
end
