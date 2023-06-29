module.exports = {
  title: `sxhxun.com`,
  description: `SEHYEON DEV-BLOG`,
  language: `ko`, // `ko`, `en` => currently support versions for Korean and English
  siteUrl: `https://www.sxhxun.com`,
  ogImage: `/og-image.png`, // Path to your in the 'static' folder
  comments: {
    utterances: {
      repo: `BBOXEEEE/blog`, // `zoomkoding/zoomkoding-gatsby-blog`,
    },
  },
  ga: '0', // Google Analytics Tracking ID
  author: {
    name: `박세현`,
    bio: {
      role: `개발자`,
      description: ['배우는 것을 좋아하는', 'JAVA를 좋아하는', '능동적으로 일하는'],
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
        date: '2023.06.29 ~',
        activity: '개발 블로그 운영',
        links: {
          post: '',
          github: 'https://github.com/BBOXEEEE/blog',
          demo: 'https://sxhxun.com',
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
    ],
  },
};
