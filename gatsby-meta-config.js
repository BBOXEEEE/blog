module.exports = {
  title: `sxhxun.com`,
  description: `개발자 박세현`,
  language: `ko`, // `ko`, `en` => currently support versions for Korean and English
  siteUrl: `https://www.sxhxun.com`,
  ogImage: `/og-image.png`, // Path to your in the 'static' folder
  comments: {
    utterances: {
      repo: `BBOXEEEE/blog`, // `zoomkoding/zoomkoding-gatsby-blog`,
    },
  },
  ga: 'G-PK033REFBV', // Google Analytics Tracking ID
  author: {
    name: `박세현`,
    bio: {
      role: `개발자`,
      description: ['배우는 것을 좋아하는', 'JAVA를 좋아하는', '매일 조금씩 성장하는'],
      thumbnail: 'thumbnail.png', // Path to the image in the 'asset' folder
    },
    social: {
      github: `https://github.com/BBOXEEEE`,
      instagram: `https://www.instagram.com/_sxhxun/`,
      email: `noeyhesx@naver.com`,
    },
  },

  // metadata for About Page
  about: {
    timestamps: [
      // =====       [Timestamp Sample and Structure]      =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!) =====
      {
        date: '',
        activity: '',
        links: {
          github: '',
          post: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        date: '2019.02 ~',
        activity: '한국기술교육대학교 컴퓨터공학부 입학',
        links: {
          post: '',
          github: '',
          demo: '',
        },
      },
      {
        date: '2023.04 ~ 2023.05',
        activity: '프로세스 스케줄링 시뮬레이터 개발',
        links: {
          post: '',
          github: 'https://github.com/BBOXEEEE/Process-Scheduling-Simulator-Web',
          demo: 'https://process-scheduler.link/simulator.html',
        },
      },
      {
        date: '2023.05 ~ 2023.06',
        activity: '헬스 커뮤니티 Light Weight Baby 개발',
        links: {
          post: '',
          github: 'https://github.com/BBOXEEEE/WebProgramming-TermProject',
          demo: 'http://lightweightbaby.kro.kr',
        },
      },
      {
        date: '2023.05 ~ 2023.06',
        activity: '무인 과일가게 가맹점주 Admin 페이지 API 개발',
        links: {
          post: '',
          github: '',
          demo: 'http://ppapfruits.kro.kr',
        },
      },
      {
        date: '2023.06.29 ~',
        activity: '개발 블로그 운영',
        links: {
          post: '',
          github: 'https://github.com/BBOXEEEE/blog',
          demo: 'https://www.sxhxun.com',
        },
      },
    ],

    projects: [
      // =====        [Project Sample and Structure]        =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!)  =====
      {
        title: '',
        description: '',
        techStack: ['', ''],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        title: '프로세스 스케줄링 시뮬레이터',
        description: 
          '2023학년도 1학기 운영체제 과목을 수강하면서 프로세스 스케줄링에 대해 배웠습니다. 팀프로젝트로 프로세스 스케줄링 시뮬레이터를 제작했습니다. 제작한 스케줄링 알고리즘은 FCFS, Round-Robin, SPN, SRTN, HRRN 그리고 커스텀 알고리즘입니다. 사용자의 입력, API 호출, 응답을 바탕으로 스케줄링 결과를 가시화하는 View 구현을 담당했습니다.',
        techStack: ['HTML', 'Javascript', 'CSS'],
        thumbnailUrl: 'scheduling-simulator.png',
        links: {
          post: '',
          github: 'https://github.com/BBOXEEEE/Process-Scheduling-Simulator-Web',
          demo: 'https://process-scheduler.link/simulator.html',
        },
      },
      {
        title: '헬스 커뮤니티 Light Weight Baby',
        description: 
          '2023학년도 1학기 웹프로그래밍 텀프로젝트로 헬스 커뮤니티를 개발했습니다. 늘어나는 자기관리에 대한 관심과 헬스인의 증가로 자유롭게 소통할 수 있고 정보를 공유할 수 있는 커뮤니티를 개발하고자 했습니다. 운동 루틴과 같은 정보를 공유하고 질의응답을 통해 자유로운 소통을 추구합니다. 쪽지 기능은 사용자 간 더욱 긴밀한 소통을 가능하게 합니다. 또한, 운동 가이드를 제공하며 추천 가이드 영상을 제공합니다. 또한, 관리자가 선정한 운동 보조식품, 용품 등 제품 추천 기능을 포함하고 있습니다.',
        techStack: ['HTML', 'Javascript', 'PHP'],
        thumbnailUrl: 'light-weight-baby.png',
        links: {
          post: '',
          github: 'https://github.com/BBOXEEEE/WebProgramming-TermProject',
          demo: 'http://lightweightbaby.kro.kr',
        },
      },
      {
        title: '무인 과일가게 가맹점주 Admin 페이지 API',
        description: 
          '2023학년도 1학기 데이터베이스설계 팀프로젝트로 무인 과일가게 가맹점주 Admin 페이지를 개발했습니다. 데이터베이스 설계와 API 설계를 맡아서 개발을 진행했습니다. 데이터베이스 설계 과정 (요구사항 분석, 개념적 설계, 논리적 설계, 물리적 설계, 정규화) 과정을 거쳐 데이터베이스를 설계 후 REST API를 제작했습니다. 주요 기능으로 가맹점주는 가맹점 추가, 상품 등록, 주문 목록 조회, 상품 목록 조회, 발주 목록 조회, 지출 목록 조회 기능이 있습니다. 소비자는 전국 가맹점들의 재고 현황을 확인하고 상품을 주문할 수 있습니다. 유통기한이 지난 상품은 Event Scheduler를 통해 매일 자정 상품 상태가 업데이트 됩니다.',
        techStack: ['nodejs', 'MySQL'],
        thumbnailUrl: 'ppapfruits.png',
        links: {
          post: '',
          github: '',
          demo: 'http://ppapfruits.kro.kr',
        },
      },
    ],
  },
};
